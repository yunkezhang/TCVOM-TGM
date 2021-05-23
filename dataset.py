import logging
import os
import json
import random
import sys
import time
from collections import OrderedDict
import cv2
import glob
import imgaug
import imgaug.augmenters as iaa
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from imgaug import parameters as iap

class VideoMattingDatasetOnline(torch.utils.data.IterableDataset):
    VIDEO_SHAPE = (1080, 1920)
    FG_FOLDER = 'FG_done'
    BG_FOLDER = 'BG_done'
    IMG_PADDING_VALUE = [103.53, 116.28, 123.675] # BGR
    A_INTER = cv2.INTER_LINEAR
    def __init__(self, data_root, video, gt_idx, image_shape):
        self.data_root = data_root
        self.image_shape = image_shape
        idx = sorted(gt_idx)
        with open(os.path.join(self.data_root, 'frame_corr.json'), 'r') as f:
            frame_corr = json.load(f)
        fnlist = [(k, frame_corr[k]) for k in sorted(frame_corr.keys()) \
            if os.path.dirname(k) == video]
        self.samples = []
        for i in idx:
            fgp = os.path.join(self.data_root, self.FG_FOLDER, fnlist[i][0])
            bgp = os.path.join(self.data_root, self.BG_FOLDER, fnlist[i][1])
            fg = np.float32(cv2.imread(fgp, cv2.IMREAD_UNCHANGED)) / 255.0
            bg = np.float32(cv2.imread(bgp)) / 255.0
            a = fg[..., -1:]
            fg = fg[..., :-1]
            img = fg * a + bg * (1. - a)
            img = np.uint8(np.clip(img, 0, 1) * 255)
            self.samples.append((img, a))

        self.shape_aug = iaa.CropToFixedSize(\
            width=self.image_shape[0], height=self.image_shape[1])

    def make_frames3(self, img, a, FM0_only=True):
        # input: fg, bg, alpha
        # output: listfg, listbg, lista
        if FM0_only:
            assert isinstance(img, (list, tuple))
            assert isinstance(a, (list, tuple))
        def _rotate_(M, center = None, angle = None, scale = None):
            M_now = cv2.getRotationMatrix2D((center[0],center[1]), angle, scale)
            M_now = np.concatenate((M_now,[[0,0,1]]))
            return np.matmul(M, M_now)

        def _shear_(M,vec, center):
            M_now = np.float32([[1,vec[0],0],[vec[1],1,0]])
            M[0,2] = -M[0,1]*center[0]/2
            M[1,2] = -M[1,0]*center[1]/2
            M_now = np.concatenate((M_now,[[0,0,1]]))
            return np.matmul(M,M_now)
        
        def _move_(M,vec):
            M_now = np.float32([[1,0,vec[0]],[0,1,vec[1]]])
            M_now = np.concatenate((M_now,[[0,0,1]]))
            return np.matmul(M,M_now)

        def _get_random_var_(w,h,
                            SHEAR_MAX = 0.5,
                            MOVE_MAX = 20,
                            ROTATE_SHEAR_MAX_CENTER = 10,
                            ROTATE_MAX_ANGLE = 2,
                            ROTATE_MIN_SCALE = 1.00,
                            ROTATE_MAX_SCALE = 1.00):
            center = ROTATE_SHEAR_MAX_CENTER*2*(np.random.random(2)-0.5)+np.array([w/2,h/2],np.float32)
            move = np.random.randint(-MOVE_MAX,MOVE_MAX,[2]) if MOVE_MAX > 0 else np.array([0, 0])
            angle = (np.random.random()-0.5)*2.*ROTATE_MAX_ANGLE
            scale = np.random.random()*(ROTATE_MAX_SCALE-ROTATE_MIN_SCALE)+ROTATE_MIN_SCALE
            shear = np.random.random(size=[2]) * SHEAR_MAX
            #print("center move angle scale")
            #print(center,move,angle,scale)
            return center,shear,angle,scale,move

        def _get_new_M(M, var):
            center, shear, angle, scale, move = var
            _rM = _rotate_(M, center, angle, scale)
            _sM = _shear_(_rM, shear, center)
            _mM = _move_(_sM, move)
            return _mM

        imgs, alphas = [None] * 3, [None] * 3
        fh, fw = img[0].shape[:2] if FM0_only else img.shape[:2]
        rh, rw = (np.random.random() - 0.5) * 2., (np.random.random() - 0.5) * 2.
        I = np.eye(3, dtype=np.float32)
        FF_var = _get_random_var_(fw, fh, \
            SHEAR_MAX=0.1,
            MOVE_MAX=0,
            ROTATE_SHEAR_MAX_CENTER=10, \
            ROTATE_MAX_ANGLE=10, \
            ROTATE_MIN_SCALE=0.9,
            ROTATE_MAX_SCALE=1.1)
        FM0 = _get_new_M(I, FF_var)
        if FM0_only:
            FM_ = [I, I, I]
        else:
            FSTEP_var = _get_random_var_(fw, fh, \
                SHEAR_MAX=0.1,
                ROTATE_SHEAR_MAX_CENTER=10, \
                ROTATE_MAX_ANGLE=10, \
                ROTATE_MIN_SCALE=0.9,
                ROTATE_MAX_SCALE=1.1
                )
            FMs = _get_new_M(I, FSTEP_var)
            I_ = np.random.randint(0, 3)
            if I_ == 0:
                FM_ = [I, FMs, np.matmul(FMs, FMs)]
            elif I_ == 1:
                FM_ = [np.linalg.inv(FMs), I, FMs]
            elif I_ == 2:
                _t = np.linalg.inv(FMs)
                FM_ = [np.matmul(_t, _t), _t, I]

        masks = []
        for i in range(3):
            FM = np.matmul(FM_[i], FM0)
            img_ = img[i] if FM0_only else img
            a_ = a[i] if FM0_only else a
            imgs[i] = cv2.warpPerspective(img_, FM, (fw, fh))
            alphas[i] = cv2.warpPerspective(a_, FM, (fw, fh), flags=self.A_INTER)[..., np.newaxis]
            mask = np.ones((fh, fw), dtype=np.float32)
            mask_idx = np.nonzero(cv2.warpPerspective(mask, FM, (fw, fh), flags=self.A_INTER) == 0)
            imgs[i][mask_idx] = self.IMG_PADDING_VALUE
            mask[mask_idx] = 0
            masks.append(mask[..., np.newaxis])

        return imgs, alphas, masks

    def get_sample(self, sample_idx):
        #use_frame = len(self.samples) > 2 and np.random.random() > 0.5
        ### HACK ###
        use_frame = False
        if use_frame:
            idx = sorted(np.random.choice(range(len(self.samples)), 3, replace=False))
            if np.random.random() > 0.5:
                idx = idx[::-1]
        good_sample = False
        while not good_sample:
            if use_frame:    
                img, a = [], []
                for i in idx:
                    img.append(np.array(self.samples[i][0]))
                    a.append(np.array(self.samples[i][1]))
            else:
                img = np.array(self.samples[sample_idx][0])
                a = np.array(self.samples[sample_idx][1])
            
            imgs, alphas, masks = self.make_frames3(img, a, FM0_only=use_frame)
            try_count = 0

            good_sample = False
            while not good_sample and try_count <= 20:
                shape_aug = self.shape_aug.to_deterministic()
                good_sample = True
                for i in range(3):
                    imgs[i] = shape_aug.augment_image(imgs[i])
                    alphas[i] = shape_aug.augment_image(alphas[i])
                    masks[i] = shape_aug.augment_image(masks[i])

                    if np.sum(np.logical_and(alphas[i] > 0, alphas[i] < 255)) < 400:
                        good_sample = False
                        break
                try_count += 1
        
        ti = np.stack(imgs)
        ta = np.stack(alphas) * 255
        tm = np.stack(masks)
        
        ti = torch.from_numpy(ti).permute(0, 3, 1, 2).float()
        ta = torch.from_numpy(ta).permute(0, 3, 1, 2).float()
        tm = torch.from_numpy(tm).permute(0, 3, 1, 2).float()
        return ti, ta, tm

    def get_iter_train(self, st, ed):
        while True:
            _id = random.randint(int(st), int(ed))
            yield self.get_sample(_id)

    def __iter__(self):
        return self.get_iter_train(0, len(self.samples) - 1)

class TestDataOnline(VideoMattingDatasetOnline):
    A_INTER = cv2.INTER_NEAREST
    def __init__(self, data_root, image_shape, gt_idx, dilation=None):
        self.data_root = data_root
        self.image_shape = image_shape
        imglist = sorted(glob.glob(os.path.join(self.data_root, '*_rgb.png')))
        trilist = sorted(glob.glob(os.path.join(self.data_root, '*_trimap.png')))
        idx = []
        if gt_idx != []:
            assert len(imglist) == len(trilist)
            for i in gt_idx:
                idx.append((imglist[i], trilist[i]))
        else:
            for _id, ip in enumerate(imglist):
                ipid = ip[:-8]
                for tp in trilist:
                    if tp.startswith(ipid):
                        idx.append((ip, tp))
        print (idx)
        
        self.samples = []
        for i in idx:
            img = np.float32(cv2.imread(i[0]))
            tri = cv2.imread(i[1], cv2.IMREAD_GRAYSCALE)
            if dilation > 0:
                trimask = np.uint8((tri>0)*(tri<255))
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation, dilation))
                trimask = cv2.dilate(trimask, kernel)
                tri = np.where(trimask != 0, 128, tri)
            tri = np.float32(tri) / 255.0
            self.samples.append((img, tri))

        self.shape_aug = iaa.CropToFixedSize(\
            width=self.image_shape[0], height=self.image_shape[1])