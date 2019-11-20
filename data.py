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
####### DEFINING THE DATASET CLASS #########
#The dataset class takes the csv file and which type of ground truth do we want the model to train on
class TumorSegmentationDataset(Dataset):
    # Read the csv file as a pandas dataframe 
    def __init__(self, csv_file, which_gt):
        self.df = pd.read_csv(csv_file)
        self.which_gt = which_gt
    def __len__(self):
        return len(self.df)
    # This function does the data augmentations on the image and ground truth by taking both of them as parameters to the function
    def transform(self,img ,gt):
    # This augmentation step is for patch extraction of the image and also performing a couple of spatial augmentations such as shear etc. 
        img, gt = augment_spatial_2(img, gt, (128,128,128))
        img, gt = img.copy(), gt.copy()
        # This is to make the data 4D from 5D and this works since our bacth size is akways one (since a batch size more than 1 doesn't fit in the memory). The dimensionality is reduced because the subsequent augmentation functions require the input to be 4D while the previous one required it to tbe 5D 
        img, gt = img[0], gt[0]        
        # The purpose of the below given augmentation functions is self explanatory from the name of the functions 
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
	# This is a gaussian noise augmentation where each of the modality is added with gaussian noise to better it's generaliation ability
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
        # This is an artifical modality that we use in addition to the other 4 modalities as it is shown to improve the performance of WT dice 
        flair_th = (flair>0.2)*(flair)
        # Concatenating the 5 modalities into a single 4D image 
        image = np.concatenate((t1,t2,t1ce,flair,flair_th),axis = 0)        
        # This is the place where we select what kind of ground truth we want to train our model with - explained in the train_parameters.cfg file 
        if gt_which == 0:
            gt = one_hot_nonoverlap(gt)
        if gt_which == 1:
            gt = one_hot_2_overlap(gt)
        image = np.expand_dims(image, axis = 0)
        gt = np.expand_dims(gt, axis = 0)
        # This is the function as explained above which does data augmentation and returns the augmented image and ground truth 
        image, gt = self.transform(image, gt)
        # Creating a dictionary named sample (which essentially means data sample) which contains the image and the corresponding ground truth
        sample = {'image': image, 'gt' : gt}
        return sample
