#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 17:07:06 2019

@author: megh
"""

import os
import glob
import numpy as np
import math as m
import nibabel as nib
import SimpleITK as sitk
from tqdm import tqdm
import sys


def bbox2_3D(img):

    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]
            

    return rmin, rmax, cmin, cmax, zmin, zmax

def crop_pad(img_array, rmin, rmax, cmin, cmax, zmin, zmax, is_mask=False):
    pu_x = pb_x = pu_y = pb_y = pu_z = pb_z = 0 
    img_array = img_array[rmin:rmax,cmin:cmax,zmin:zmax]
    dim = img_array.shape
    dimen = dim[0]
    if dimen%16 != 0:
        x = 16*(m.trunc(dimen/16)+1)
        pu_x = m.trunc((x - dimen)/2)
        pb_x = x - dimen - pu_x
        
    dimen = dim[1]
    
    if dimen%16 != 0:
        x = 16*(m.trunc(dimen/16)+1)
        pu_y = m.trunc((x - dimen)/2)
        pb_y = x - dimen - pu_y
        
        
    dimen = dim[2]
    
    if dimen%16 != 0:
        x = 16*(m.trunc(dimen/16)+1)
        pu_z = m.trunc((x - dimen)/2)
        pb_z = x - dimen - pu_z

    cropped_padded_image_data = np.pad(img_array,((pu_x, pb_x),(pu_y,pb_y),(pu_z,pb_z)),'constant')
    
    if not is_mask:
        normalize(cropped_padded_image_data)
        
    return cropped_padded_image_data

def normalize(image_array):
    temp = image_array > 0
    temp_image_array = image_array[temp]    
    mu = np.mean(temp_image_array)
    sig = np.std(temp_image_array)
    image_array[temp] = (image_array[temp] - mu)/sig
    return image_array

def preprocess_patient(patient_path):
       
    os.chdir(patient_path)
    
    t1_path = glob.glob("*t1.nii.gz")
    t1_path = str(t1_path[0])
    t1_data = nib.load(patient_path + t1_path)
    t1_affine = t1_data.affine
    t1_data = t1_data.get_fdata()
    
    flair_path = glob.glob("*flair.nii.gz")
    flair_path = str(flair_path[0])
    flair_data =  nib.load(patient_path + "/" + flair_path)
    flair_affine = flair_data.affine
    flair_data = flair_data.get_fdata()
    
    t1ce_path = glob.glob("*t1ce.nii.gz")
    t1ce_path = str(t1ce_path[0])
    t1ce_data =  nib.load(patient_path + "/" + t1ce_path)
    t1ce_affine = t1ce_data.affine
    t1ce_data = t1ce_data.get_fdata()
    
    t2_path = glob.glob("*t2.nii.gz")
    t2_path = str(t2_path[0])
    t2_data = nib.load(patient_path + "/" + t2_path)
    t2_affine = t2_data.affine
    t2_data = t2_data.get_fdata()

    seg_path = glob.glob("*seg.nii.gz")
    seg_path = str(seg_path[0])
    seg_data = nib.load(patient_path + "/" + seg_path)
    seg_affine = seg_data.affine
    seg_data = seg_data.get_fdata()

    
    # Get the rmin, rmax, cmin, cmax, zmin, zmax for all modalities 
    (rmin_t1, rmax_t1, cmin_t1, cmax_t1, zmin_t1, zmax_t1) = bbox2_3D(t1_data)
    (rmin_t2, rmax_t2, cmin_t2, cmax_t2, zmin_t2, zmax_t2) = bbox2_3D(t2_data)
    (rmin_flair, rmax_flair, cmin_flair, cmax_flair, zmin_flair, zmax_flair) = bbox2_3D(flair_data)
    (rmin_t1ce, rmax_t1ce, cmin_t1ce, cmax_t1ce, zmin_t1ce, zmax_t1ce) = bbox2_3D(t1ce_data)
    (rmin_seg, rmax_seg, cmin_seg, cmax_seg, zmin_seg, zmax_seg) = bbox2_3D(seg_data)   

    rmin = min(rmin_t1, rmin_t2, rmin_flair, rmin_t1ce,rmin_seg)
    rmax = max(rmax_t1, rmax_t2, rmax_flair, rmax_t1ce,rmax_seg)
    cmin = min(cmin_t1, cmin_t2, cmin_flair, cmin_t1ce,cmin_seg)
    cmax = max(cmax_t1, cmax_t2, cmax_flair,cmax_t1ce,cmax_seg)
    zmin = min(zmin_t1, zmin_t2, zmin_flair, zmin_t1ce,zmin_seg)
    zmax = max(zmax_t1, zmax_t2, zmax_flair, zmax_t1ce,zmax_seg)
    
    # Crop_pad all modalities
    t1_data =  crop_pad(t1_data, rmin, rmax, cmin, cmax, zmin, zmax, is_mask=False)
    flair_data = crop_pad(flair_data, rmin, rmax, cmin, cmax, zmin, zmax, is_mask=False)
    t2_data = crop_pad(t2_data, rmin, rmax, cmin, cmax, zmin, zmax, is_mask=False)
    t1ce_data = crop_pad(t1ce_data, rmin, rmax, cmin, cmax, zmin, zmax, is_mask=False)
    seg_data = crop_pad(seg_data, rmin, rmax, cmin, cmax, zmin, zmax, is_mask=True)

    return t1_data,t1_affine,flair_data,flair_affine,t2_data,t2_affine,t1ce_data,t1ce_affine,seg_data,seg_affine

#Path of the raw data
path_data = "/cbica/home/bhaleram/comp_space/brats/BraTS_2019_Test/" 
#Path to save the preprocessed patient images
path = "/cbica/home/bhaleram/comp_space/brats/data/test/"
a = os.listdir(path_data)
for file in tqdm(a):
    (t1_data,t1_affine,flair_data,flair_affine,t2_data,t2_affine,t1ce_data,t1ce_affine,seg_data,seg_affine) = preprocess_patient(path_data + file + "/")
    #below givrn is the directory where you want to save the preprocessed images
    os.mkdir(path + file)
    new_image = nib.Nifti1Image(t1_data, affine = t1_affine)
    nib.save(new_image,path + file + "/" + file + "_t1.nii.gz")
    new_image = nib.Nifti1Image(flair_data, affine = flair_affine)
    nib.save(new_image,path + file + "/" +file + "_flair.nii.gz")
    new_image = nib.Nifti1Image(t2_data, affine = t2_affine)
    nib.save(new_image,path + file + "/" + file + "_t2.nii.gz")
    new_image = nib.Nifti1Image(t1ce_data, affine = t1ce_affine)
    nib.save(new_image,path + file + "/" + file + "_t1ce.nii.gz")
    new_image = nib.Nifti1Image(seg_data, affine = seg_affine)
    nib.save(new_image,path + file + "/" + file + "_seg.nii.gz")
    print("Iteration Done")


