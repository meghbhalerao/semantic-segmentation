#This code tries to implement the post processing step of Fabian Isenne's paper  - the objective of this script is to get the threshold value individually for every experiment 
import torch 
import numpy as np
from data import TumorSegmentationDataset
from torch.utils.data import Dataset, DataLoader
from losses import *
import os 
import nibabel as nib
#Path of the train segmentations
seg_pred_path = "/cbica/home/bhaleram/comp_space/brats/Segmentation_Labels_Train/seg_labels29/seg_majority_voting/"
seg_gt_path = "/cbica/home/bhaleram/comp_space/brats/Brats2019_Original_Data/BraTS2019/Train_Original_All/"

#This is the voxel threshold for the segmentation generation 
thresh = 1500 
#A quantity similar to the learning rate used to update the threshold measured in the number of voxels
lam = 1000
#list of ET voxel values 
ET_voxel_vals = []
num_et_voxels = 0
def convert_to_3D(seg):
    seg = seg[0,:,:,:]*4 + seg[1,:,:,:]*1 + seg[2,:,:,:]*2 + seg[3,:,:,:]*0
    return seg
def ET_dice(seg_pred,seg_gt):
    ET_pred = (seg_pred==4).astype(int)
    ET_gt = (seg_gt==4).astype(int)
    pred_flat = ET_pred.flatten()
    gt_flat = ET_gt.flatten()
    dice = (2*sum(pred_flat*gt_flat) + 1e-7)/(sum(pred_flat)+sum(gt_flat))
    return dice    

diff_loss = 0

patient_list = os.listdir(seg_gt_path)

for patient in patient_list:
    seg_pred = nib.load(seg_pred_path + patient + ".nii.gz").get_fdata().astype(int)
    seg_gt = nib.load(seg_gt_path + patient + "/" + patient + "_seg.nii.gz").get_fdata().astype(int)
    dice = ET_dice(seg_pred,seg_gt)
    print("The enhancing tumor dice is:", dice)
    if dice < 1e-7:
        num_et_voxels = sum((seg_pred==4).astype(int).flatten())
        ET_voxel_vals.append(num_et_voxels)
ET_voxel_vals = np.array(ET_voxel_vals)
avg_num_ET_voxels = sum(ET_voxel_vals)/len(ET_voxel_vals)  
print(avg_num_voxels)
