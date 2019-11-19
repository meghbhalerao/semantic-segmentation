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
    # set the modalities here
    #modalities = ['t1','t1ce','t2','flair']
    
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
    
    t1_bfc_path = glob.glob("*t1_bfc.nii.gz")
    t1_bfc_path = str(t1_bfc_path[0])
    t1_bfc_data = nib.load(patient_path + t1_bfc_path)
    t1_bfc_affine = t1_bfc_data.affine
    t1_bfc_data = t1_bfc_data.get_fdata()
    
    t1ce_bfc_path = glob.glob("*t1ce_bfc.nii.gz")
    t1ce_bfc_path = str(t1ce_bfc_path[0])
    t1ce_bfc_data = nib.load(patient_path + t1ce_bfc_path)
    t1ce_bfc_affine = t1ce_bfc_data.affine
    t1ce_bfc_data = t1ce_bfc_data.get_fdata()
    

    flair_bfc_path = glob.glob("*flair_bfc.nii.gz")
    flair_bfc_path = str(flair_bfc_path[0])
    flair_bfc_data = nib.load(patient_path + flair_bfc_path)
    flair_bfc_affine = flair_bfc_data.affine
    flair_bfc_data = flair_bfc_data.get_fdata()
    
    t2_bfc_path = glob.glob("*t2_bfc.nii.gz")
    t2_bfc_path = str(t2_bfc_path[0])
    t2_bfc_data = nib.load(patient_path + t2_bfc_path)
    t2_bfc_affine = t2_bfc_data.affine
    t2_bfc_data = t2_bfc_data.get_fdata()
    
    
    # Get the rmin, rmax, cmin, cmax, zmin, zmax for all modalities 
    (rmin_t1, rmax_t1, cmin_t1, cmax_t1, zmin_t1, zmax_t1) = bbox2_3D(t1_data)
    (rmin_t2, rmax_t2, cmin_t2, cmax_t2, zmin_t2, zmax_t2) = bbox2_3D(t2_data)
    (rmin_flair, rmax_flair, cmin_flair, cmax_flair, zmin_flair, zmax_flair) = bbox2_3D(flair_data)
    (rmin_t1ce, rmax_t1ce, cmin_t1ce, cmax_t1ce, zmin_t1ce, zmax_t1ce) = bbox2_3D(t1ce_data)
    (rmin_seg, rmax_seg, cmin_seg, cmax_seg, zmin_seg, zmax_seg) = bbox2_3D(seg_data)
    (rmin_t1_bfc, rmax_t1_bfc, cmin_t1_bfc, cmax_t1_bfc, zmin_t1_bfc, zmax_t1_bfc) = bbox2_3D(t1_bfc_data)
    (rmin_t1ce_bfc, rmax_t1ce_bfc, cmin_t1ce_bfc, cmax_t1ce_bfc, zmin_t1ce_bfc, zmax_t1ce_bfc) = bbox2_3D(t1ce_bfc_data)
    (rmin_flair_bfc, rmax_flair_bfc, cmin_flair_bfc, cmax_flair_bfc, zmin_flair_bfc, zmax_flair_bfc) = bbox2_3D(flair_bfc_data)
    (rmin_t2_bfc, rmax_t2_bfc, cmin_t2_bfc, cmax_t2_bfc, zmin_t2_bfc, zmax_t2_bfc) = bbox2_3D(t2_bfc_data)



    
    rmin = min(rmin_t1, rmin_t2, rmin_flair, rmin_t1ce, rmin_seg, rmin_t1_bfc, rmin_t1ce_bfc, rmin_flair_bfc, rmin_t2_bfc)
    rmax = max(rmax_t1, rmax_t2, rmax_flair, rmax_t1ce, rmax_seg, rmax_t1_bfc, rmax_t1ce_bfc, rmax_flair_bfc, rmax_t2_bfc)
    cmin = min(cmin_t1, cmin_t2, cmin_flair, cmin_t1ce, cmin_seg, cmin_t1_bfc, cmin_t1ce_bfc, cmin_flair_bfc, cmin_t2_bfc)
    cmax = max(cmax_t1, cmax_t2, cmax_flair,cmax_t1ce, cmax_seg, cmax_t1_bfc, cmax_t1ce_bfc, cmax_flair_bfc, cmax_t2_bfc)
    zmin = min(zmin_t1, zmin_t2, zmin_flair, zmin_t1ce, zmin_seg, zmin_t1_bfc, zmin_t1ce_bfc, zmin_flair_bfc, zmin_t2_bfc)
    zmax = max(zmax_t1, zmax_t2, zmax_flair, zmax_t1ce, zmax_seg, zmax_t1_bfc, zmax_t1ce_bfc, zmax_flair_bfc, zmax_t2_bfc)
    
    # Crop_pad all modalities
    t1_data =  crop_pad(t1_data, rmin, rmax, cmin, cmax, zmin, zmax, is_mask=False)
    flair_data = crop_pad(flair_data, rmin, rmax, cmin, cmax, zmin, zmax, is_mask=False)
    t2_data = crop_pad(t2_data, rmin, rmax, cmin, cmax, zmin, zmax, is_mask=False)
    t1ce_data = crop_pad(t1ce_data, rmin, rmax, cmin, cmax, zmin, zmax, is_mask=False)
    seg_data =  crop_pad(seg_data, rmin, rmax, cmin, cmax, zmin, zmax, is_mask=True)
    t1_bfc_data =  crop_pad(t1_bfc_data, rmin, rmax, cmin, cmax, zmin, zmax, is_mask=False)
    t1ce_bfc_data =  crop_pad(t1ce_bfc_data, rmin, rmax, cmin, cmax, zmin, zmax, is_mask=False)
    flair_bfc_data =  crop_pad(flair_bfc_data, rmin, rmax, cmin, cmax, zmin, zmax, is_mask=False)
    t2_bfc_data =  crop_pad(t2_bfc_data, rmin, rmax, cmin, cmax, zmin, zmax, is_mask=False)



    
        
    
    return t1_data,t1_affine,flair_data,flair_affine,t2_data,t2_affine,t1ce_data,t1ce_affine,seg_data,seg_affine, t1_bfc_data, t1_bfc_affine, t1ce_bfc_data,t1ce_bfc_affine,flair_bfc_data, flair_bfc_affine, t2_bfc_data, t2_bfc_affine



path_data = "/home/megh/cbica_server/home/brats/data1/LGG/" 
#path where u wanna save the preporcessed patients
path = "/home/megh/work/pp/"
#each patient folder is assumed to contain - t1,t2,t1ce,flair,seg,t1bfc,t1cebfc,flairbfc - if it doesnt contain any 1 of the before mentioned images just run the script and later
#we can delete the unecessary thing 
a = os.listdir(path_data)
for file in tqdm(a):
    (t1_data,t1_affine,flair_data,flair_affine,t2_data,t2_affine,t1ce_data,t1ce_affine,seg_data,seg_affine, t1_bfc_data, t1_bfc_affine, t1ce_bfc_data,t1ce_bfc_affine,flair_bfc_data, flair_bfc_affine, t2_bfc_data, t2_bfc_affine) = preprocess_patient(path_data + file + "/")
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
    new_image = nib.Nifti1Image(t1_bfc_data, affine = t1_bfc_affine)
    nib.save(new_image,path + file + "/" + file + "_t1_bfc.nii.gz")  
    new_image = nib.Nifti1Image(t1ce_bfc_data, affine = t1ce_bfc_affine)
    nib.save(new_image,path + file + "/" + file + "_t1ce_bfc.nii.gz") 
    new_image = nib.Nifti1Image(flair_bfc_data, affine = flair_bfc_affine)
    nib.save(new_image,path + file + "/" + file + "_flair_bfc.nii.gz") 
    new_image = nib.Nifti1Image(t2_bfc_data, affine = t2_bfc_affine)
    nib.save(new_image,path + file + "/" + file + "_t2_bfc.nii.gz") 
