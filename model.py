from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models
 
# general libs
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import time
import tqdm
import os
import argparse
import copy
import sys

from helpers import *
 
class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim and stride==1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
 
        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)
 
 
    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))
 
        if self.downsample is not None:
            x = self.downsample(x)
         
        return x + r 

class Encoder_M(nn.Module):
    def __init__(self):
        super(Encoder_M, self).__init__()
        self.conv1_m = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_o = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 # 1/4, 256
        self.res3 = resnet.layer2 # 1/8, 512
        self.res4 = resnet.layer3 # 1/8, 1024

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, in_f, in_m, in_o):
        f = (in_f - self.mean) / self.std
        m = torch.unsqueeze(in_m, dim=1).float() # add channel dim
        o = torch.unsqueeze(in_o, dim=1).float() # add channel dim

        x = self.conv1(f) + self.conv1_m(m) + self.conv1_o(o) 
        x = self.bn1(x)
        c1 = self.relu(x)   # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)   # 1/4, 256
        r3 = self.res3(r2) # 1/8, 512
        r4 = self.res4(r3) # 1/8, 1024
        return r4, r3, r2, c1, f
 
class Encoder_Q(nn.Module):
    def __init__(self):
        super(Encoder_Q, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 # 1/4, 256
        self.res3 = resnet.layer2 # 1/8, 512
        self.res4 = resnet.layer3 # 1/8, 1024

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, in_f):
        f = (in_f - self.mean) / self.std

        x = self.conv1(f) 
        x = self.bn1(x)
        c1 = self.relu(x)   # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)   # 1/4, 256
        r3 = self.res3(r2) # 1/8, 512
        r4 = self.res4(r3) # 1/8, 1024
        return r4, r3, r2, c1, f


class Refine(nn.Module):
    def __init__(self, inplanes, planes, scale_factor=2):
        super(Refine, self).__init__()
        self.convFS = nn.Conv2d(inplanes, planes, kernel_size=(3,3), padding=(1,1), stride=1)
        self.ResFS = ResBlock(planes, planes)
        self.ResMM = ResBlock(planes, planes)
        self.scale_factor = scale_factor

    def forward(self, f, pm):
        s = self.ResFS(self.convFS(f))
        m = s + F.interpolate(pm, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        m = self.ResMM(m)
        return m

class Decoder(nn.Module):
    def __init__(self, mdim):
        super(Decoder, self).__init__()
        self.convFM = nn.Conv2d(1024, mdim, kernel_size=(3,3), padding=(1,1), stride=1)
        self.ResMM = ResBlock(mdim, mdim)
        self.RF3 = Refine(512, mdim) # 1/8 -> 1/4
        self.RF2 = Refine(256, mdim) # 1/4 -> 1

        self.pred2 = nn.Conv2d(mdim, 2, kernel_size=(3,3), padding=(1,1), stride=1)

    def forward(self, r4, r3, r2):
        m4 = self.ResMM(self.convFM(r4))
        m3 = self.RF3(r3, m4) # out: 1/8, 256
        m2 = self.RF2(r2, m3) # out: 1/4, 256

        p2 = self.pred2(F.relu(m2))
        
        p = F.interpolate(p2, scale_factor=4, mode='bilinear', align_corners=False)
        return p #, p2, p3, p4



class Memory(nn.Module):
    def __init__(self):
        super(Memory, self).__init__()
 
    def forward(self, m_in, m_out, q_in, q_out):  # m_in: o,c,t,h,w
        B, D_e, T, H, W = m_in.size()
        _, D_o, _, _, _ = m_out.size()

        mi = m_in.view(B, D_e, T*H*W) 
        mi = torch.transpose(mi, 1, 2)  # b, THW, emb
 
        qi = q_in.view(B, D_e, H*W)  # b, emb, HW
 
        p = torch.bmm(mi, qi) # b, THW, HW
        p = p / math.sqrt(D_e)
        p = F.softmax(p, dim=1) # b, THW, HW

        mo = m_out.view(B, D_o, T*H*W) 
        mem = torch.bmm(mo, p) # Weighted-sum B, D_o, HW
        mem = mem.view(B, D_o, H, W)

        mem_out = torch.cat([mem, q_out], dim=1)

        return mem_out, p

class KeyValue(nn.Module):
    # Not using location
    def __init__(self, indim, keydim, valdim):
        super(KeyValue, self).__init__()
        self.Key = nn.Conv2d(indim, keydim, kernel_size=(3,3), padding=(1,1), stride=1)
        self.Value = nn.Conv2d(indim, valdim, kernel_size=(3,3), padding=(1,1), stride=1)
 
    def forward(self, x):  
        return self.Key(x), self.Value(x)




class STM(nn.Module):
    def __init__(self):
        super(STM, self).__init__()
        self.Encoder_M = Encoder_M() 
        self.Encoder_Q = Encoder_Q() 

        self.KV_M_r4 = KeyValue(1024, keydim=128, valdim=512)
        self.KV_Q_r4 = KeyValue(1024, keydim=128, valdim=512)

        self.Memory = Memory()
        self.Decoder = Decoder(256)
 
    def Pad_memory(self, mems, num_objects, K):
        pad_mems = []
        for mem in mems:
                                      #  1, K, C,             1,    H,              W
            pad_mem = ToCuda(torch.zeros(1, K, mem.size()[1], 1, mem.size()[2], mem.size()[3]))
            #       -  K              C, H,W
            pad_mem[0,1:num_objects+1,:,0] = mem    # [N, C, H, W]
            pad_mems.append(pad_mem)
        return pad_mems

    def memorize(self, frame, masks, num_objects): 
        # memorize a frame 
        num_objects = num_objects[0].item()
        B, K, H, W = masks.shape # B = 1

        (frame, masks), pad = pad_divide_by([frame, masks], 16, (frame.size()[2], frame.size()[3]))

        # make batch arg list
        B_list = {'f':[], 'm':[], 'o':[]}
        for o in range(1, num_objects+1): # 1 - no
            B_list['f'].append(frame)
            B_list['m'].append(masks[:,o])
            B_list['o'].append( (torch.sum(masks[:,1:o], dim=1) + \
                torch.sum(masks[:,o+1:num_objects+1], dim=1)).clamp(0,1) )

        # make Batch
        B_ = {}
        for arg in B_list.keys():
            B_[arg] = torch.cat(B_list[arg], dim=0)

        r4, _, _, _, _ = self.Encoder_M(B_['f'], B_['m'], B_['o'])
        k4, v4 = self.KV_M_r4(r4) # num_objects, 128 and 512, H/16, W/16
        k4, v4 = self.Pad_memory([k4, v4], num_objects=num_objects, K=K)
        return k4, v4

    def Soft_aggregation(self, ps, K):
        num_objects, H, W = ps.shape
        em = ToCuda(torch.zeros(1, K, H, W)) 
        em[0,0] =  torch.prod(1-ps, dim=0) # bg prob
        em[0,1:num_objects+1] = ps # obj prob
        em = torch.clamp(em, 1e-7, 1-1e-7)
        logit = torch.log((em /(1-em)))
        return logit

    def segment(self, frame, keys, values, num_objects): 
        num_objects = num_objects[0].item()
        _, K, keydim, T, H, W = keys.shape # B = 1
        # pad
        [frame], pad = pad_divide_by([frame], 16, (frame.size()[2], frame.size()[3]))

        r4, r3, r2, _, _ = self.Encoder_Q(frame)
        k4, v4 = self.KV_Q_r4(r4)   # 1, dim, H/16, W/16
        
        # expand to ---  no, c, h, w
        k4e, v4e = k4.expand(num_objects,-1,-1,-1), v4.expand(num_objects,-1,-1,-1) 
        r3e, r2e = r3.expand(num_objects,-1,-1,-1), r2.expand(num_objects,-1,-1,-1)
        
        # memory select kv:(1, K, C, T, H, W)
        m4, viz = self.Memory(keys[0,1:num_objects+1], values[0,1:num_objects+1], k4e, v4e)
        logits = self.Decoder(m4, r3e, r2e)
        ps = F.softmax(logits, dim=1)[:,1] # no, h, w  
        #ps = indipendant possibility to belong to each object
        
        logit = self.Soft_aggregation(ps, K) # 1, K, H, W

        if pad[2]+pad[3] > 0:
            logit = logit[:,:,pad[2]:-pad[3],:]
        if pad[0]+pad[1] > 0:
            logit = logit[:,:,:,pad[0]:-pad[1]]

        return logit    

    def forward(self, *args, **kwargs):
        if args[1].dim() > 4: # keys
            return self.segment(*args, **kwargs)
        else:
            return self.memorize(*args, **kwargs)


class FullModel(nn.Module):
    def __init__(self, dilate_kernel=0, eps=0, ignore_label=255, unknown_weight=5.0):
        super(FullModel, self).__init__()
        self.DILATION_KERNEL = dilate_kernel
        self.EPS = eps
        self.IMG_SCALE = 1./255
        self.register_buffer('IMG_MEAN', torch.tensor([0.485, 0.456, 0.406]).reshape([1, 1, 3, 1, 1]).float())
        self.register_buffer('IMG_STD', torch.tensor([0.229, 0.224, 0.225]).reshape([1, 1, 3, 1, 1]).float())
        self.model = STM()
        self.ignore_label = ignore_label
        self.LOSS = nn.CrossEntropyLoss(weight=torch.tensor([1, unknown_weight, 1]).float(), ignore_index=ignore_label)

    def make_trimap(self, alpha):
        b = alpha.shape[0]
        alpha = torch.where(alpha < self.EPS, torch.zeros_like(alpha), alpha)
        alpha = torch.where(alpha > 1 - self.EPS, torch.ones_like(alpha), alpha)
        trimasks = ((alpha > 0) & (alpha < 1.)).float().split(1)
        trimaps = [None] * b
        for i in range(b):
            # trimap width: 1 - 51
            kernel_rad = int(torch.randint(0, 26, size=())) \
                if self.DILATION_KERNEL is None else self.DILATION_KERNEL
            trimaps[i] = F.max_pool2d(trimasks[i].squeeze(0), kernel_size=kernel_rad*2+1, stride=1, padding=kernel_rad)
        trimap = torch.stack(trimaps)
        # 0: bg, 1: un, 2: fg
        trimap1 = torch.where(trimap > 0.5, torch.ones_like(alpha), 2 * alpha).long()
        trimap3 = F.one_hot(trimap1.squeeze(2), num_classes=3).permute(0, 1, 4, 2, 3)
        return trimap3.float()

    def preprocess(self, a, fg, bg):
        # Data preprocess
        with torch.no_grad():
            scaled_gts = a * self.IMG_SCALE
            scaled_fgs = fg.flip([2]) * self.IMG_SCALE
            scaled_bgs = bg.flip([2]) * self.IMG_SCALE
            scaled_imgs = scaled_fgs * scaled_gts + scaled_bgs * (1. - scaled_gts)
            scaled_tris = self.make_trimap(scaled_gts)
            #alphas, features = [None] * self.sample_length, [None] * self.sample_length
            imgs = ((scaled_imgs - self.IMG_MEAN) / self.IMG_STD)#.split(1, dim=1)
        return scaled_imgs, scaled_fgs, scaled_bgs, scaled_gts, scaled_tris, imgs

    def _forward(self, imgs, tris, masks=None, og_shape=None):
        batch_size, sample_length = imgs.shape[:2]
        num_object = torch.tensor([2]).to(torch.cuda.current_device())
        GT = tris.split(1, dim=0)                               # [1, S, C, H, W]
        FG = imgs.split(1, dim=0)                               # [1, S, C, H, W]
        if masks is not None:
            M = masks.squeeze(2).split(1, dim=1)
        E = []
        E_logits = []
        # we split batch here since the original code only supports b=1
        for b in range(batch_size):
            Fs = FG[b].split(1, dim=1)                          # [1, 1, C, H, W]
            GTs = GT[b].split(1, dim=1)                         # [1, 1, C, H, W]
            Es = [GTs[0].squeeze(1)] + [None] * (sample_length - 1) # [1, C, H, W]
            ELs = []
            for t in range(1, sample_length):
                # memorize
                prev_key, prev_value = self.model(Fs[t-1].squeeze(1), Es[t-1], num_object)

                if t-1 == 0: # 
                    this_keys, this_values = prev_key, prev_value # only prev memory
                else:
                    this_keys = torch.cat([keys, prev_key], dim=3)
                    this_values = torch.cat([values, prev_value], dim=3)
                
                # segment
                logit = self.model(Fs[t].squeeze(1), this_keys, this_values, num_object)
                ELs.append(logit)
                Es[t] = F.softmax(logit, dim=1)
                
                # update
                keys, values = this_keys, this_values
            E.append(torch.cat(Es, dim=0))  # cat t
            E_logits.append(torch.cat(ELs, dim=0))

        pred = torch.stack(E, dim=0)  # stack b
        E_logits = [None] + list(torch.stack(E_logits).split(1, dim=1))
        GT = torch.argmax(tris, dim=2).split(1, dim=1)
        # Loss & Vis
        losses = []
        for t in range(1, sample_length):
            gt = GT[t].squeeze(1)
            p = E_logits[t].squeeze(1)
            if og_shape is not None:
                for b in range(batch_size):
                    h, w = og_shape[b]
                    gt[b, h:] = self.ignore_label
                    gt[b, :, w:] = self.ignore_label
            if masks is not None:
                mask = M[t].squeeze(1)
                gt = torch.where(mask == 0, torch.ones_like(gt) * self.ignore_label, gt)
            losses.append(self.LOSS(p, gt))
        loss = sum(losses) / float(len(losses))
        return pred, loss

    def forward(self, a, fg, bg, og_shape=None):
        scaled_imgs, _, _, scaled_gts, tris, imgs = self.preprocess(a, fg, bg)

        pred, loss = self._forward(imgs, tris, og_shape=og_shape)
        
        return [loss,
                scaled_imgs, pred, tris, scaled_gts]

class FullModelFinetune(FullModel):
    def preprocess(self, a, img):
        # Data preprocess
        with torch.no_grad():
            scaled_gts = a * self.IMG_SCALE
            scaled_imgs = img.flip([2]) * self.IMG_SCALE
            scaled_tris = self.make_trimap(scaled_gts)
            #alphas, features = [None] * self.sample_length, [None] * self.sample_length
            imgs = ((scaled_imgs - self.IMG_MEAN) / self.IMG_STD)#.split(1, dim=1)
        return scaled_imgs, scaled_gts, scaled_tris, imgs

    def forward(self, a, img, mask):
        scaled_imgs, scaled_gts, tris, imgs = self.preprocess(a, img)

        pred, loss = self._forward(imgs, tris, mask)
        
        return [loss,
                scaled_imgs, pred, tris, scaled_gts]