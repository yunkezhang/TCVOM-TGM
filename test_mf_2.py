import torch
from torch.autograd import Variable
from torch.utils import data

import torch.nn as nn
import torch.nn.functional as F

# general libs
import cv2
import numpy as np
import math
import multiprocessing as mp
import os
import argparse
import json
import pickle
from tqdm import tqdm
import glob

from model import STM
from config import get_cfg_defaults

class Processor():
    IMG_MEAN = np.reshape(np.array([0.485, 0.456, 0.406]), (1, 1, 3))
    IMG_STD = np.reshape(np.array([0.229, 0.224, 0.225]), (1, 1, 3))
    def __init__(self, video_shape, dilation=0):
        self.video_shape = video_shape
        self.image_shape = (int(np.ceil(video_shape[0] / 32.0) * 32),
                            int(np.ceil(video_shape[1] / 32.0) * 32))
        self.ph, self.pw = self.image_shape[0] - video_shape[0], self.image_shape[1] - video_shape[1]
        if dilation > 0:
            self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation, dilation))
        else:
            self.kernel = None

    def image(self, ip):
        img = np.float32(cv2.imread(ip)) / 255.0
        assert img.shape[:2] == self.video_shape
        img = (img - self.IMG_MEAN) / self.IMG_STD
        img = np.pad(img, ((0, self.ph), (0, self.pw), (0, 0)), mode='reflect')

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img

    def trimap(self, tp):
        gt = cv2.imread(tp, cv2.IMREAD_GRAYSCALE)
        if self.kernel is not None:
            mask = np.uint8((gt>0)*(gt<255))
            mask = cv2.dilate(mask, self.kernel)
            gt = np.where(mask != 0, 128, gt)
        gt = gt[..., np.newaxis]
        gt = np.concatenate([np.uint8(gt==0), np.uint8((gt>0)*(gt<255)), np.uint8(gt==255)], axis=-1)
        gt = np.pad(gt, ((0, self.ph), (0, self.pw), (0, 0)), mode='reflect')
        gt = torch.from_numpy(gt).permute(2, 0, 1).float()
        return gt

class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, fnlist, processor, part=None):
        self.samples = fnlist

        if part is not None:
            sp, ep, reverse, _ = part
            if sp < 0:
                self.samples = self.samples[ep::-1] if reverse else self.samples[:ep]
            else:
                self.samples = self.samples[ep:sp:-1] if reverse else self.samples[sp:ep]
            self.fidx = list(range(ep, sp, -1)) if reverse else list(range(sp, ep))
        else:
            self.fidx = list(range(len(samples)))
        self.processor = processor
        self.gt = self.processor.trimap(self.samples[0][:-8] + '_trimap.png')

    def __len__(self):
        return (len(self.samples))

    def __getitem__(self, idx):
        ip = self.samples[idx]
        return self.processor.image(ip), self.gt, idx
        

def pred(videos, device, args):
    video = videos[0]
    torch.cuda.set_device(device)

    outdir = os.path.join(args.save, video+'_mem{}'.format(args.mem_every))
    os.makedirs(outdir, exist_ok=True)

    ########################## init net
    model = STM()

    ########################## load weight if specified
    dct = torch.load(args.load, map_location=torch.device('cpu'))
    missing_keys, unexpected_keys = model.load_state_dict(dct, strict=False)
    print ('Missing keys: ' + str(sorted(missing_keys)))
    print ('Unexpected keys: ' + str(sorted(unexpected_keys)))
    print('Model loaded from', args.load)
    model.to(device)
    model.eval()

    ########################## initilization
    num_objects = 2
    fnlist = sorted(glob.glob(os.path.join(args.data, '*_rgb.png')))
    num_frames = len(fnlist)
    print (video, num_frames)
    h, w = cv2.imread(fnlist[0]).shape[:2]
    # GT_IDX: {0...num_frames-1}
    gt_idx = args.gt_idx
    parts = []
    conf = []
    # if there's a head
    if gt_idx[0] != 0:
        parts.append((-1, gt_idx[0], True, gt_idx[0]))
    for i in range(len(gt_idx)-1):
        parts.append((gt_idx[i], gt_idx[i+1], False, None))
        parts.append((gt_idx[i], gt_idx[i+1], True, None))
    # if there's a tail
    if gt_idx[-1] != num_frames-1:
        parts.append((gt_idx[-1], num_frames, False, gt_idx[-1]))
    #preds = [[] for i in range(num_frames)]

    # read all gt and fix memory
    processor = Processor((h, w), dilation=args.dilation)
    
    for part_id, part in enumerate(parts):
        gt_keys, gt_values = [], []
        part_gt_idx = [part[-1]] if part[-1] is not None else \
            [part[0], part[1]]
        with torch.no_grad():
            for gt in part_gt_idx:
                img = fnlist[gt]
                tri = fnlist[gt][:-8]+'_trimap.png'
                img, tri = processor.image(img), processor.trimap(tri)
                k, v = model(img.unsqueeze(0).to(device), tri.unsqueeze(0).to(device), torch.tensor([num_objects])) 
                gt_keys.append(k)
                gt_values.append(v)
            gt_keys = torch.cat(gt_keys, dim=3)
            gt_values = torch.cat(gt_values, dim=3)
        fixed_mem = gt_keys.shape[3]

        dataset = EvalDataset(fnlist, processor, part=part)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.n_threads,
            pin_memory=True,
            drop_last=False,
        )
        # if parts[-1] is None:
        #     conf.append({gt_idx[i]: 1.0, gt_idx[i+1]: 1.0})
        # else:
        #     conf.append({part[-1]: 1.0})
        conf.append({})
        part_len = len(dataset)

        if args.mem_every:
            mem_every = args.mem_every
        elif args.max_memory:
            mem_every = math.ceil(num_frames / float(args.max_memory))
            assert len(np.arange(0, num_frames, step=mem_every)) <= args.max_memory
        else:
            raise NotImplementedError

        to_memorize = [int(i) for i in np.arange(0, num_frames, step=mem_every)]

        ### evaluation
        print ('Start evaluation process...')
        with torch.no_grad():
            for i, dp in enumerate(dataloader):
                idx = dp[2].item()
                fidx = dataset.fidx[idx]
                curdir = os.path.join(outdir, 'part{}'.format(part_id))
                os.makedirs(curdir, exist_ok=True)
                outfn = os.path.join(curdir, os.path.basename(fnlist[fidx][:-8]+'_tri.png'))
                print (outfn)
                conf[part_id][os.path.basename(fnlist[fidx][:-8])] = 1.0 - float(i) / float(part_len)
                if i == 0:
                    last_frame, last_pred = dp[0], dp[1]
                    pred = np.uint8(last_pred[0].permute(1, 2, 0).numpy() * 255)[:h, :w]
                    #pred = cv2.hconcat([pred, pred])
                    cv2.imwrite(outfn, pred)
                    #preds[fidx].append(last_pred[0, :, :h, :w].permute(1, 2, 0).numpy())
                    last_frame = last_frame.to(device)
                    last_pred = last_pred.to(device)
                    continue
                else:
                    cur_frame = dp[0].to(device)
                    #gt = np.uint8(dp[1][0].permute(1, 2, 0).numpy() * 255)[:h, :w]

                # memorize
                if i == 1:
                    this_keys, this_values = gt_keys.clone(), gt_values.clone() # only prev memory
                else:
                    prev_key, prev_value = model(last_frame, last_pred, torch.tensor([num_objects])) 
                    this_keys = torch.cat([keys, prev_key], dim=3)
                    this_values = torch.cat([values, prev_value], dim=3)
                
                # segment
                logit = model(cur_frame, this_keys, this_values, torch.tensor([num_objects]))
                cur_pred = F.softmax(logit, dim=1)
                pred = np.uint8((cur_pred[0].permute(1, 2, 0).detach().cpu().numpy()) * 255)[:h, :w]
                #preds[fidx].append(cur_pred[0, :, :h, :w].permute(1, 2, 0).cpu().numpy())
                #output = cv2.hconcat([pred, gt])
                cv2.imwrite(outfn, pred)
                
                # update
                if i-1 in to_memorize:
                    keys, values = this_keys, this_values
                    if args.max_memory:
                        key_length = keys.shape[3]
                        if key_length == args.max_memory + fixed_mem:
                            # delete the oldest memory
                            keys = torch.cat([keys[:, :, :, :fixed_mem], \
                                             keys[:, :, :, fixed_mem+1:]], dim=3)
                            values = torch.cat([values[:, :, :, :fixed_mem], \
                                                values[:, :, :, fixed_mem+1:]], dim=3)
                last_frame = cur_frame
                last_pred = cur_pred

    with open(os.path.join(outdir, 'conf.json'), 'w') as f:
        json.dump(conf, f)

    # for i in range(num_frames):
    #     pred = np.mean(np.stack(preds[i]), axis=0)
    #     pred = np.uint8(pred * 255)
    #     outfn = os.path.splitext(os.path.basename(fnlist[i]))[0] + '_tri.png'
    #     cv2.imwrite(os.path.join(outdir, outfn), pred)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True)
    parser.add_argument('--load', help='resume from a checkpoint with optimizer parameter attached')
    parser.add_argument('--n_threads', default=4)
    parser.add_argument('--save', required=True)
    parser.add_argument('--max_memory', type=int, default=None)
    parser.add_argument('--mem_every', type=int, default=5)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('opts',
                    help="Modify config options using the command-line",
                    default=None,
                    nargs=argparse.REMAINDER)

    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    return args, cfg

def main(args, cfg):
    device = torch.device('cuda:{}'.format(args.gpu))
    # if args.save is None:
    #     args.save = 'vmd_results/{}'.format(os.path.splitext(os.path.basename(args.load))[0])
    args.data = os.path.normpath(cfg.DATASET.PATH)
    #args.gt_idx = None if cfg.FINETUNE.GT_IDX == [] else sorted(cfg.FINETUNE.GT_IDX)
    if cfg.FINETUNE.GT_IDX == []:
        args.gt_idx = []
        imglist = sorted(glob.glob(os.path.join(args.data, '*_rgb.png')))
        trilist = sorted(glob.glob(os.path.join(args.data, '*_trimap.png')))
        for _id, ip in enumerate(imglist):
            ipid = ip[:-8]
            for tp in trilist:
                if tp.startswith(ipid):
                    args.gt_idx.append(_id)
    else:
        args.gt_idx = sorted(cfg.FINETUNE.GT_IDX)
    args.dilation = cfg.FINETUNE.DILATION
    video = [os.path.basename(args.data)]
    print ('Video:', video)
    print ('GT idx:', args.gt_idx)
    pred(video, device, args)


if __name__ == "__main__":
    args, cfg = parse_args()
    main(args, cfg)