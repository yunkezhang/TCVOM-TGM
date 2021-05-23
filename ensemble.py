import argparse
import os, sys
import glob
import numpy as np
import cv2 as cv
from tqdm import tqdm
import json

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predroot', required=True)
    parser.add_argument('--dataroot', required=True)
    parser.add_argument('videos', nargs='*')
    args = parser.parse_args()
    return args         

if __name__ == "__main__":
    args = parse()
    if args.videos == []:
        dirs = [i for i in sorted(os.listdir(args.predroot)) \
            if os.path.isdir(os.path.join(args.predroot, i))]
    else:
        dirs = args.videos
    data_root = args.dataroot
    
    for v in tqdm(dirs, ascii=True):
        if v[0] == '.': continue
        v = os.path.normpath(v)
        with open(os.path.join(args.predroot, v, 'conf.json'), 'r') as f:
            conf = json.load(f)
        parts = list(range(len(conf)))
        ids = {}
        for p in parts:
            fnlist = [os.path.join(args.predroot, v, 'part{}'.format(p), i+'_tri.png') \
                      for i in sorted(conf[p].keys())]
            for fn in fnlist:
                _id = os.path.basename(fn)[:-8]
                if _id not in ids:
                    ids[_id] = [(fn, conf[p][_id])]
                else:
                    ids[_id].append((fn, conf[p][_id]))
        h,w = cv.imread(ids[_id][0][0]).shape[:2]
        outdir = os.path.join(args.predroot, v, 'ensemble')
        visdir = os.path.join(args.predroot, v, 'envis')
        os.makedirs(visdir, exist_ok=True)
        os.makedirs(outdir, exist_ok=True)
        for i in tqdm(sorted(ids.keys()), ascii=True):
            if '_mem' in v:
                rgbpath = os.path.join(data_root, os.path.basename(v[:v.rfind('_')]), i+'_rgb.png')
            else:
                rgbpath = os.path.join(data_root, os.path.basename(v), i+'_rgb.png')
            rgb = cv.imread(rgbpath)
            outfn = os.path.join(outdir, i+'_trimap.png')
            res = np.zeros((h, w, 3))
            lbl = np.zeros((h, w), dtype=np.uint8)
            if len(ids[i]) == 1:
                res += cv.imread(ids[i][0][0])
            else:
                for pair in ids[i]:
                    fn, conf = pair
                    res += cv.imread(fn) * conf
            res = np.argmax(res, axis=-1)
            lbl = np.where(res==1, 128, np.uint8(res / 2.0 * 255))
            vis = np.float32(rgb) * 0.5 + np.float32(lbl[..., np.newaxis]) * 0.5
            vis = np.uint8(np.clip(vis, 0, 255))
            cv.imwrite(outfn, lbl)
            cv.imwrite(os.path.join(visdir, i+'.png'), vis)
