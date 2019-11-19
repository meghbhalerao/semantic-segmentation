import numpy as np
import torch 
import os 
import nibabel as nib

path = "/cbica/comp_space/bhaleram/brats/Segmentation_Labels_Test/seg_labels29/seg_majority_voting/"
save_path = "/cbica/comp_space/bhaleram/Segmentation_Labels_Test/seg_labels29/seg_post_process/"
#this value of the threshold is obtained from running the other script on the already generated segmentations
thresh = 1000
patient_list = os.listdir(path)
for patient in patient_list:
    seg = nib.load(path+patient)
    aff = seg.affine
    seg_data = seg.get_fdata()
    num_et = sum(((seg_data==4).astype(int)).flatten())
    et_mask = (seg_data==4).astype(int)
    ed_mask = (seg_data==2).astype(int)
    nc_mask = (seg_data==1).astype(int)
    if num_et < thresh:
        nc_mask = nc_mask + et_mask
        et_mask = et_mask*0
    seg_pp = et_mask*4 + ed_mask*2 + nc_mask*1
    seg_pp = nib.NifTI1Image(seg_pp,aff)
    nib.save(seg_pp,save_path+patient)
    
