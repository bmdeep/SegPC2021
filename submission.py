
#!/bin/python

import os
import glob
import cv2
import numpy as np
import argparse
import platform
import tqdm

def coords(mask,sz,type_):
    sz_ = mask.shape[:2]
    if sz[0] != sz_[0] or sz[1] != sz_[1]:
        mask = cv2.resize(mask.astype(np.uint8),(sz[1],sz[0]),interpolation=cv2.INTER_NEAREST).astype(np.bool)
    x,y = np.where(mask)
    return ';'.join([','.join([str(i),str(j)]) for i,j in zip(x,y)])

parser = argparse.ArgumentParser(description="add the srouce directory path")
parser.add_argument('-s','--source',help="provide path of source directory")
parser.add_argument('-d','--destination',help="kindly provide the destination submission dir path")
parser.add_argument('-nu','--nucleus',help="the label assigned to nucleus in the instance mask",type=int,default=40)
parser.add_argument('-cy','--cyto',help="the label assigned to cytoplasm in the instance mask",type=int,default=20)

args = parser.parse_args()

source = args.source
dest = args.destination

sz = (1080,1440)


delim = '\\' if platform.system() == 'Windows' else '/'


if source[-1] != delim:
    source += delim


imgs = list(set([i.split(delim)[-1].split('_')[0] for i in glob.glob(source+'/*')]))
imgs = sorted(imgs)
submission = []

for img in tqdm.tqdm(imgs):
    insts = glob.glob(source+img+'_*')
    res = [img.split(delim)[-1]]
    for ins in insts:
        ins_mask = cv2.imread(ins,0)
        ins_nu = coords(ins_mask == args.nucleus,sz,'n')
        ins_cy = coords(ins_mask == args.cyto,sz,'c')
        res.append(ins_nu+' '+ins_cy)
    submission.append('\t'.join(res))

with open(dest+'submission.txt','w') as fp:
    print(len(submission))
    fp.write('\n'.join(submission))
    print("saved: "+dest+"submission.txt")


