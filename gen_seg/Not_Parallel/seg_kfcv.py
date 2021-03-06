#/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 10:02:35 2019

@author: megh
"""

import numpy as np
import torch
import nibabel as nib
import os
import sys
import pandas as pd


data_path = "/cbica/home/bhaleram/comp_space/brats/BraTS_2019_Validation/"
save_path = "/cbica/home/bhaleram/comp_space/brats/Segmentation_Labels_Test/seg_labels5/seg_average/"
model_path1 = "/cbica/home/bhaleram/comp_space/brats/model/5/mod1.pt"
model_path2 = "/cbica/home/bhaleram/comp_space/brats/model/5/mod2.pt"
model_path3 = "/cbica/home/bhaleram/comp_space/brats/model/5/mod3.pt"
model_path4 = "/cbica/home/bhaleram/comp_space/brats/model/5/mod4.pt"
model_path5 = "/cbica/home/bhaleram/comp_space/brats/model/5/mod5.pt"

patient_name = sys.argv[1]


t2 = np.expand_dims(nib.load(data_path + patient_name + "/" + patient_name + "_t2.nii.gz").get_fdata(), axis = 0)
t1ce = np.expand_dims(nib.load(data_path + patient_name + "/" + patient_name + "_t1ce.nii.gz").get_fdata(), axis = 0)
flair = np.expand_dims(nib.load(data_path + patient_name + "/" + patient_name + "_flair.nii.gz").get_fdata(), axis = 0)
t1 = np.expand_dims(nib.load(data_path + patient_name + "/" + patient_name + "_t1.nii.gz").get_fdata(), axis = 0)


img = np.expand_dims(np.concatenate((t1,t2,t1ce,flair),axis = 0), axis = 0)    
img = torch.tensor(img)
aff = nib.load(data_path + patient_name + "/" + patient_name + "_t1.nii.gz").affine

model1 = torch.load(model_path1, map_location = 'cpu')
model2 = torch.load(model_path2, map_location = 'cpu')
model3 = torch.load(model_path3, map_location = 'cpu')
model4 = torch.load(model_path4, map_location = 'cpu')
model5 = torch.load(model_path5, map_location = 'cpu')


model1.cpu()
model2.cpu()
model3.cpu()
model4.cpu()
model5.cpu()

model1.eval()
model2.eval()
model3.eval()
model4.eval()
model5.eval()

(batch, channel, x, y, z) = img.shape

def convert_to_3D(seg):
    seg = seg[0,:,:,:]*4 + seg[1,:,:,:]*1 + seg[2,:,:,:]*2 + seg[3,:,:,:]*0
    return seg
psize = 128
master_seg = np.zeros((x,y,z), dtype = float)

torch.cuda.empty_cache()
seg_pred1 = model1(img.float()) 
seg_pred2 = model2(img.float())
seg_pred3 = model3(img.float())
seg_pred4 = model4(img.float())
seg_pred5 = model5(img.float())

seg_pred = seg_pred1 + seg_pred2 + seg_pred3 + seg_pred4 + seg_pred5

seg = seg_pred.cpu().detach().numpy()  
seg = (seg>2.5).astype(int)
seg = seg[0]
seg = convert_to_3D(seg)
master_seg[0:x,0:y,0:z] = seg
print("Iteration done")

seg = master_seg
df = pd.read_csv("info1.csv")
pid = patient_name
patient_info = df.loc[df['PatientID'] == pid]
Rdim = 240
Cdim = 240
Zdim = 155
Rmin = patient_info.iloc[0,1]
Rmax = patient_info.iloc[0,2]
Cmin = patient_info.iloc[0,3]
Cmax = patient_info.iloc[0,4]
Zmin = patient_info.iloc[0,5]
Zmax = patient_info.iloc[0,6]

PRmin = patient_info.iloc[0,7]
PRmax = patient_info.iloc[0,8]
PCmin = patient_info.iloc[0,9]
PCmax = patient_info.iloc[0,10]
PZmin = patient_info.iloc[0,11]
PZmax = patient_info.iloc[0,12]
(a,b,c) = seg.shape
seg = seg[PRmin:a - PRmax, PCmin:b - PCmax,PZmin:c - PZmax]
seg_new = np.zeros((300,300,300))

(a,b,c) = seg.shape
seg_new[Rmin:Rmin+a, Cmin:Cmin+b, Zmin:Zmin+c] = seg

seg_new = seg_new[0:240, 0:240, 0:155]

seg_new = nib.Nifti1Image(seg_new, aff)

nib.save(seg_new, save_path + patient_name + ".nii.gz")
