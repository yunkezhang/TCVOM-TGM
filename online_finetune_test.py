import argparse
import logging
import os
import shutil
import time
import timeit
import shutil

import numpy as np
import cv2 as cv
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as torch_dist
import torch.nn.functional as F
from torch import nn, optim
from torch.utils import data
from torchvision.utils import save_image
from tqdm import tqdm

from config import get_cfg_defaults
from dataset import TestDataOnline
from model import FullModelFinetune
from helpers import AverageMeter, create_logger, \
    torch_barrier, reduce_tensor, poly_lr, const_lr

def write_image(outdir, out, step, max_batch=4):
    with torch.no_grad():
        scaled_imgs, pred, tris, scaled_gts = out
        b, s, _, h, w = scaled_imgs.shape
        b = max_batch if b > max_batch else b
        save_image(scaled_imgs[:max_batch].reshape(b*s, 3, h, w), os.path.join(outdir, 'vis_image_{}.png'.format(step)), nrow=s)
        save_image(tris[:max_batch].reshape(b*s, 3, h, w), os.path.join(outdir, 'vis_tris_{}.png'.format(step)), nrow=s)
        save_image(pred[:max_batch].reshape(b*s, 3, h, w), os.path.join(outdir, 'vis_preds_{}.png'.format(step)), nrow=s)
        save_image(scaled_gts[:max_batch].reshape(b*s, 1, h, w), os.path.join(outdir, 'vis_as_{}.png'.format(step)), nrow=s)

def train(epoch, train_iter, steps_per_val, base_lr,
          total_epochs, optimizer, model, 
          adjust_learning_rate, print_freq, 
          image_freq, image_outdir):    
    # Training
    # STM DISABLES BN DURING TRAINING
    model.eval()
    
    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*steps_per_val
    for i_iter in range(steps_per_val):
        def handle_batch():
            img, a, mask = next(train_iter)      # [B, 3, 3 or 1, H, W]
            out = model(a, img, mask)
            loss = out[0].mean()

            model.zero_grad()
            loss.backward()
            optimizer.step()
            return loss.detach(), out[1:]

        loss, vis_out = handle_batch()

        reduced_loss = reduce_tensor(loss)
        # measure elapsed time
        batch_time = time.time() - tic
        tic = time.time()

        # update average loss
        ave_loss.update(reduced_loss.item())
        torch_barrier()

        adjust_learning_rate(optimizer,
                            base_lr,
                            total_epochs * steps_per_val,
                            i_iter+cur_iters)

        if i_iter % print_freq == 0:
            msg = 'Iter:[{}/{}], Time: {:.2f}, '.format(\
                i_iter+cur_iters, total_epochs * steps_per_val, batch_time)
            msg += 'lr: {}, Avg. Loss: {:.6f} | Current: Loss: {:.6f}, '.format(
                [x['lr'] for x in optimizer.param_groups],
                ave_loss.average(), ave_loss.value())
            logging.info(msg)
        
        if i_iter % image_freq == 0:
            write_image(image_outdir, vis_out, i_iter+cur_iters)

def get_sampler(dataset, shuffle=True):
    if torch_dist.is_available() and torch_dist.is_initialized():
        from torch.utils.data.distributed import DistributedSampler
        return DistributedSampler(dataset, shuffle=shuffle)
    else:
        return None

def main(cfg_name, cfg, device):
    torch.cuda.set_device(device)
    cfg_name += cfg.SYSTEM.EXP_SUFFIX
    random_seed = cfg.SYSTEM.RANDOM_SEED
    #assert local_rank >= 0
    load_ckpt = cfg.TRAIN.LOAD_CKPT
    load_opt = cfg.TRAIN.LOAD_OPT
    base_lr = cfg.TRAIN.BASE_LR
    weight_decay = cfg.TRAIN.WEIGHT_DECAY
    output_dir = cfg.SYSTEM.OUTDIR
    start = timeit.default_timer()
    # cudnn related setting
    cudnn.benchmark = cfg.SYSTEM.CUDNN_BENCHMARK
    cudnn.deterministic = cfg.SYSTEM.CUDNN_DETERMINISTIC
    cudnn.enabled = cfg.SYSTEM.CUDNN_ENABLED
    if random_seed > 0:
        import random
        print('Seeding with', random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)

    logger, final_output_dir = create_logger(output_dir, cfg_name, 'train')
    print (cfg)
    with open(os.path.join(final_output_dir, 'config.yaml'), 'w') as f:
        f.write(str(cfg))
    image_outdir = os.path.join(final_output_dir, 'training_images')
    os.makedirs(os.path.join(final_output_dir, 'training_images'), exist_ok=True)

    # build model
    model = FullModelFinetune(eps=0, \
        unknown_weight=cfg.TRAIN.UNKNOWN_WEIGHT)
    torch_barrier()

    # prepare data
    train_dataset = TestDataOnline(
        data_root=cfg.DATASET.PATH,
        gt_idx=cfg.FINETUNE.GT_IDX,
        image_shape=cfg.TRAIN.TRAIN_INPUT_SIZE,
        dilation=cfg.FINETUNE.DILATION
    )
    train_sampler = get_sampler(train_dataset)
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU,
        num_workers=cfg.SYSTEM.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler)
    train_iter = iter(trainloader)

    if load_ckpt != '':
        dct = torch.load(load_ckpt, map_location=torch.device('cpu'))
        missing_keys, unexpected_keys = model.model.load_state_dict(dct, strict=False)
        logger.info('Missing keys: ' + str(sorted(missing_keys)))
        logger.info('Unexpected keys: ' + str(sorted(unexpected_keys)))
        logger.info("=> loaded checkpoint from {}".format(load_ckpt))

    model = torch.nn.DataParallel(model, device_ids=[device])

    # optimizer
    params_dict = {k: v for k, v in model.named_parameters() \
        if v.requires_grad and 'bn' not in k}
        
    params_count = 0
    logging.info('=> Parameters needs to be optimized:')
    for k in sorted(params_dict):
        logging.info('\t=> {}, size: {}'.format(k, list(params_dict[k].size())))
        params_count += params_dict[k].shape.numel()
    logging.info('=> Total Parameters: {}'.format(params_count))
        
    params = [{'params': list(params_dict.values()), 'lr': base_lr}]
    optimizer = torch.optim.Adam(params, lr=base_lr, weight_decay=weight_decay)
    adjust_lr = const_lr

    if load_opt != '':
        optimizer.load_state_dict(torch.load(load_opt, map_location='cpu'))
        start_epoch = int(load_opt[load_opt.rfind('_')+1:-8])
    else:
        start_epoch = 0

    total_steps = cfg.TRAIN.TOTAL_STEPS
    steps_per_val = cfg.FINETUNE.SAVE_FREQ
    if steps_per_val == -1:
        steps_per_val = total_steps
    assert total_steps % steps_per_val == 0
    total_epochs = total_steps // steps_per_val
    print_freq = cfg.TRAIN.PRINT_FREQ
    image_freq = cfg.TRAIN.IMAGE_FREQ
   
    #validate(testloader, model, len(test_dataset), local_rank)
    for epoch in range(total_epochs):
        train(epoch, train_iter, steps_per_val, base_lr, total_epochs,
              optimizer, model, adjust_lr, print_freq, image_freq, \
              image_outdir)

        weight_fn = os.path.join(final_output_dir,\
            'checkpoint_{}.pth.tar'.format(epoch+1))
        logger.info('=> saving checkpoint to {}'.format(weight_fn))
        torch.save(model.module.model.state_dict(), weight_fn)
        
    end = timeit.default_timer()
    logger.info('Time: %d sec.' % np.int((end-start)))
    logger.info('Done')

def parse_args():
    parser = argparse.ArgumentParser(description='Train network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument("--gpu", type=int, default=0)
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


if __name__ == "__main__":
    args, cfg = parse_args()
    main(os.path.splitext(os.path.basename(args.cfg))[0], cfg, torch.device('cuda:{}'.format(args.gpu)))
