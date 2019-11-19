from augmentations.augs import *
from augmentations.color_aug import *
from augmentations.noise_aug import *
from augmentations.spatial_augs import *
from augmentations.utils import *
import nibabel as nib
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import os
import random
from all_augmentations import *
from utils import *
import random
class TumorSegmentationDataset(Dataset):
    def __init__(self, csv_file, which_gt):
        self.df = pd.read_csv(csv_file)
        self.which_gt = which_gt
    def __len__(self):
        return len(self.df)
    # This function does data augmentation 
    def transform(self,img ,gt):
        img, gt = augment_spatial_2(img, gt, (128,128,128))
        img, gt = img.copy(), gt.copy()
        img, gt = img[0], gt[0]        
        if random.random()<0.12:
            img, gt = augment_rot90(img, gt)
            img, gt = img.copy(), gt.copy()        
        if random.random()<0.12:
            img, gt = augment_mirroring(img, gt)
            img, gt = img.copy(), gt.copy()
        if random.random()<0.12:
            img, gt = augment_rotate_angle(img, gt, 45)
            img, gt = img.copy(), gt.copy() 
        if random.random()<0.12:
            img, gt = augment_rotate_angle(img, gt, 180)
            img, gt = img.copy(), gt.copy() 
        if random.random() < 0.12:
            img, gt = np.fliplr(img).copy(), np.fliplr(gt).copy()
        if random.random() < 0.12:
            img[1] = gaussian(img[1],True,0,0.1)   
            img[2] = gaussian(img[2],True,0,0.1)
            img[3] = gaussian(img[3],True,0,0.1)
            img[0] = gaussian(img[0],True,0,0.1)
            img[4] = gaussian(img[4],True,0,0.1)

        return img,gt
    # This function is not needed here right now, but this function is used to extract random patches of psize from the image
    def rcrop(self,imshape,psize):        
        xshift = random.randint(0,imshape[0]-psize)
        yshift = random.randint(0,imshape[1]-psize)
        zshift = random.randint(0,imshape[2]-psize)
        return xshift, yshift, zshift
    # Function which returns the data point to be passed into the neural network 
    def __getitem__(self, index):
    # Getting the path of each of the image modalities by accessing items from the pandas dataframe 
        t1_path = self.df.iloc[index, 0]
        t2_path = self.df.iloc[index,2]
        t1ce_path = self.df.iloc[index,3]
        flair_path = self.df.iloc[index,5]
        gt_which = self.which_gt
        gt_path = self.df.iloc[index,7]
    # Loading each of the nifti image as a numpy array using nibabel package 
        gt = nib.load(gt_path).get_fdata()
        t1 = nib.load(t1_path).get_fdata()
        t2 = nib.load(t2_path).get_fdata()
        t1ce = nib.load(t1ce_path).get_fdata()
        flair = nib.load(flair_path).get_fdata()
     # Expanding the dimension of each of the image so that they can be concatenated to form a 4D image   
        t1 = np.expand_dims(t1,axis = 0)
        t2 = np.expand_dims(t2,axis = 0)
        t1ce = np.expand_dims(t1ce,axis = 0)
        flair = np.expand_dims(flair,axis = 0)
        
        flair_th = (flair>0.2)*(flair)
        
        image = np.concatenate((t1,t2,t1ce,flair,flair_th),axis = 0)        
         
        if gt_which == 0:
            gt = one_hot_nonoverlap(gt)
        if gt_which == 1:
            gt = one_hot_2_overlap(gt)
        image = np.expand_dims(image, axis = 0)
        gt = np.expand_dims(gt, axis = 0)
        
        image, gt = self.transform(image, gt)
        
        sample = {'image': image, 'gt' : gt}
        return sample
