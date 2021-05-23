from __future__ import division
#torch
import torch
from torch.autograd import Variable
from torch.utils import data

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models
import torch.distributed as torch_dist

# general libs
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import time
import os
import copy
import logging
from pathlib import Path
from importlib import reload

def ToCuda(xs):
    if torch.cuda.is_available():
        if isinstance(xs, list) or isinstance(xs, tuple):
            return [x.cuda() for x in xs]
        else:
            return xs.cuda() 
    else:
        return xs


def pad_divide_by(in_list, d, in_size):
    out_list = []
    h, w = in_size
    if h % d > 0:
        new_h = h + d - h % d
    else:
        new_h = h
    if w % d > 0:
        new_w = w + d - w % d
    else:
        new_w = w
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pad_array = (int(lw), int(uw), int(lh), int(uh))
    for inp in in_list:
        out_list.append(F.pad(inp, pad_array))
    return out_list, pad_array



def overlay_davis(image,mask,colors=[255,0,0],cscale=2,alpha=0.4):
    """ Overlay segmentation on top of RGB image. from davis official"""
    # import skimage
    from scipy.ndimage.morphology import binary_erosion, binary_dilation

    colors = np.reshape(colors, (-1, 3))
    colors = np.atleast_2d(colors) * cscale

    im_overlay = image.copy()
    object_ids = np.unique(mask)

    for object_id in object_ids[1:]:
        # Overlay color on  binary mask
        foreground = image*alpha + np.ones(image.shape)*(1-alpha) * np.array(colors[object_id])
        binary_mask = mask == object_id

        # Compose image
        im_overlay[binary_mask] = foreground[binary_mask]

        # countours = skimage.morphology.binary.binary_dilation(binary_mask) - binary_mask
        countours = binary_dilation(binary_mask) ^ binary_mask
        # countours = cv2.dilate(binary_mask, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))) - binary_mask
        im_overlay[countours,:] = 0

    return im_overlay.astype(image.dtype)


def torch_barrier():
    if torch_dist.is_available() and torch_dist.is_initialized():
        torch_dist.barrier()

def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that 
    ALL PROCESSES has the averaged results.
    """
    if torch_dist.is_initialized():
        world_size = torch_dist.get_world_size()
        if world_size < 2:
            return inp
        with torch.no_grad():
            reduced_inp = inp
            torch.distributed.all_reduce(reduced_inp)
            torch.distributed.barrier()
        return reduced_inp / world_size
    return inp

def print_loss_dict(loss, save=None):
    s = ''
    for key in sorted(loss.keys()):
        s += '{}: {:.6f}\n'.format(key, loss[key])
    print (s)
    if save is not None:
        with open(save, 'w') as f:
            f.write(s)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

def create_logger(output_dir, cfg_name, phase='train'):
    root_output_dir = Path(output_dir)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    final_output_dir = root_output_dir / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    # reset logging
    logging.shutdown()
    reload(logging)
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger, str(final_output_dir)

def poly_lr(optimizer, base_lr, max_iters, cur_iters, power=0.9):
    lr = base_lr*((1-float(cur_iters)/max_iters)**(power))
    optimizer.param_groups[0]['lr'] = lr
    return lr

def const_lr(optimizer, base_lr, max_iters, cur_iters):
    optimizer.param_groups[0]['lr'] = base_lr
    return base_lr

def stair_lr(optimizer, base_lr, max_iters, cur_iters):
    #         0, 40,  80,  120,   160
    ratios = [1, 0.5, 0.1, 0.05, 0.01]
    progress = cur_iters / float(max_iters)
    if progress < 0.2:
        ratio = ratios[0]
    elif progress < 0.4:
        ratio = ratios[1]
    elif progress < 0.6:
        ratio = ratios[2]
    elif progress < 0.8:
        ratio = ratios[3]
    else:
        ratio = ratios[-1]
    lr = base_lr * ratio
    optimizer.param_groups[0]['lr'] = lr
    return lr

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

STR_DICT = {
    'poly': poly_lr,
    'const': const_lr,
    'stair': stair_lr
}