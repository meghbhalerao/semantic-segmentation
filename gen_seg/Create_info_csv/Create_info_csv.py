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
import csv

def bbox2_3D(img):

    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]
            

    return rmin, rmax, cmin, cmax, zmin, zmax

def crop_pad(img_array, rmin, rmax, cmin, cmax, zmin, zmax):
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
    
    return pu_x, pb_x, pu_y, pb_y, pu_z, pb_z
    
def preprocess_patient(patient_path, patient):

    t1_data = nib.load(patient_path + patient + "_t1.nii.gz")
    t1_data = t1_data.get_fdata()

    flair_data =  nib.load(patient_path + patient + "_flair.nii.gz")
    flair_data = flair_data.get_fdata()

    t1ce_data =  nib.load(patient_path + patient + "_t1ce.nii.gz")
    t1ce_data = t1ce_data.get_fdata()
    
    t2_data = nib.load(patient_path + patient + "_t2.nii.gz")
    t2_data = t2_data.get_fdata()

    seg_data = nib.load(patient_path + patient + "_seg.nii.gz")
    seg_data = seg_data.get_fdata()
    
    
    # Get the rmin, rmax, cmin, cmax, zmin, zmax for all modalities 
    (rmin_t1, rmax_t1, cmin_t1, cmax_t1, zmin_t1, zmax_t1) = bbox2_3D(t1_data)
    (rmin_t2, rmax_t2, cmin_t2, cmax_t2, zmin_t2, zmax_t2) = bbox2_3D(t2_data)
    (rmin_flair, rmax_flair, cmin_flair, cmax_flair, zmin_flair, zmax_flair) = bbox2_3D(flair_data)
    (rmin_t1ce, rmax_t1ce, cmin_t1ce, cmax_t1ce, zmin_t1ce, zmax_t1ce) = bbox2_3D(t1ce_data)
    (rmin_seg, rmax_seg, cmin_seg, cmax_seg, zmin_seg, zmax_seg) = bbox2_3D(seg_data)

    
    rmin = min(rmin_t1, rmin_t2, rmin_flair, rmin_t1ce, rmin_seg)
    rmax = max(rmax_t1, rmax_t2, rmax_flair, rmax_t1ce, rmax_seg) 
    cmin = min(cmin_t1, cmin_t2, cmin_flair, cmin_t1ce, cmin_seg)
    cmax = max(cmax_t1, cmax_t2, cmax_flair,cmax_t1ce, cmax_seg)
    zmin = min(zmin_t1, zmin_t2, zmin_flair, zmin_t1ce, zmin_seg)
    zmax = max(zmax_t1, zmax_t2, zmax_flair, zmax_t1ce, zmax_seg)
    
    # Crop_pad all modalities
    return rmin, rmax, cmin, cmax, zmin, zmax

f1 = open('train_info.csv','w+')
path_data ="/cbica/home/bhaleram/comp_space/brats/Brats2019_Original_Data/BraTS2019/Train_Original_All/" 
patient_list = os.listdir(path_data)
f1.write('PatientID,Rmin,Rmax,Cmin,Cmax,Zmin,Zmax,PRmin,PRmax,PCmin,PCmax,PZmin,PZmax \n')
f1 = open('train_info.csv','a')
for patient in patient_list:
    patient_path =  path_data + patient
    rmin, rmax, cmin, cmax, zmin, zmax = preprocess_patient(patient_path + "/", patient)
    t1_data = nib.load(patient_path + "/" + patient + "_t1.nii.gz")
    t1_data = t1_data.get_fdata()
    pu_x, pb_x, pu_y, pb_y, pu_z, pb_z = crop_pad(t1_data,rmin, rmax, cmin, cmax, zmin, zmax)
    print(rmin, rmax, cmin, cmax, zmin, zmax,pu_x, pb_x, pu_y, pb_y, pu_z, pb_z) 
    f1 = open('train_info.csv','a')
    with f1:
        w = csv.writer(f1)
        w.writerow([patient,rmin,rmax,cmin,cmax,zmin,zmax,pb_x,pu_x,pb_y,pu_y,pb_z,pu_z])
        


