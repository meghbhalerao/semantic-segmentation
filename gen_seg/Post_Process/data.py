import nibabel as nib
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import os
import random
from utils import *
import random
class TumorSegmentationDataset(Dataset):
    def __init__(self, csv_file, which_gt):
        self.df = pd.read_csv(csv_file)
        self.which_gt = which_gt
    def __len__(self):
        return len(self.df)

    def rcrop(self,imshape,psize):        
        xshift = random.randint(0,imshape[0]-psize)
        yshift = random.randint(0,imshape[1]-psize)
        zshift = random.randint(0,imshape[2]-psize)
        return xshift, yshift, zshift

    def __getitem__(self, index):
        t1_path = self.df.iloc[index, 0]
        t2_path = self.df.iloc[index,2]
        t1ce_path = self.df.iloc[index,3]
        flair_path = self.df.iloc[index,5]
        gt_which = self.which_gt
        gt_path = self.df.iloc[index,7]
        #Input all nifti images as numpy arrays
        t1 = nib.load(t1_path).get_fdata()
        t2 = nib.load(t2_path).get_fdata()
        t1ce = nib.load(t1ce_path).get_fdata()
        flair = nib.load(flair_path).get_fdata()
        gt = nib.load(gt_path).get_fdata()

        
        psize = 128
        shp = t1.shape
        
        xsh, ysh, zsh = self.rcrop(shp,psize)
        t1 = t1[xsh:xsh+psize,ysh:ysh+psize,zsh:zsh+psize]
        t2 = t2[xsh:xsh+psize,ysh:ysh+psize,zsh:zsh+psize]
        t1ce = t1ce[xsh:xsh+psize,ysh:ysh+psize,zsh:zsh+psize]
        flair = flair[xsh:xsh+psize,ysh:ysh+psize,zsh:zsh+psize]
        gt = gt[xsh:xsh+psize,ysh:ysh+psize,zsh:zsh+psize]

        t1 = np.expand_dims(t1,axis = 0)
        t2 = np.expand_dims(t2,axis = 0)
        t1ce = np.expand_dims(t1ce,axis = 0)
        flair = np.expand_dims(flair,axis = 0)
        image = np.concatenate((t1,t2,t1ce,flair),axis = 0)        

        if gt_which == 0:
            gt = one_hot_nonoverlap(gt)
        if gt_which == 1:
            gt = one_hot_2_overlap(gt)
        #image = np.expand_dims(image, axis = 0)
        #gt = np.expand_dims(gt, axis = 0)
        #print(image.shape, gt.shape)
        sample = {'image': image, 'gt' : gt}
        return sample
