#This code tries to implement the post processing step of Fabian Isenne's paper  - the objective of this script is to get the threshold value individually for every experiment 
import torch 
import numpy as np
from data import TumorSegmentationDataset
from torch.utils.data import Dataset, DataLoader
from losses import *
#Model can either be region based training or can be non-overlap trained 
model_path = "/cbica/comp_space/bhaleram/brats/model/14/mod.pt"
#This is the voxel threshold for the segmentation generation 
thresh = 1500 
#A quantity similar to the learning rate used to update the threshold measured in the number of voxels
lam = 1000

def convert_to_3D(seg):
    seg = seg[0,:,:,:]*4 + seg[1,:,:,:]*1 + seg[2,:,:,:]*2 + seg[3,:,:,:]*0
    return seg
diff_loss = 0
    
model = torch.load(model_path)
model.cuda()
model.eval
num_epochs = 4
num_classes = 4

dataset_train = TumorSegmentationDataset("train.csv", 0)
train_loader = DataLoader(dataset_train,batch_size= 1,shuffle=True, num_workers=4)
for ep in range(num_epochs):
    for batch_idx, (subject) in enumerate(train_loader):
        image = subject['image']
        mask = subject['gt']
        
        image, mask = image.float().cuda(), mask.float().cuda()
        output = model(image)
        seg = output.cpu().detach().numpy()
        
        seg = (seg>0.5).astype(int)  
        num_et = seg[:,0,:,:,:].sum()
        print(num_et)
        seg = torch.tensor(seg).cuda().float()
        loss_prev = MCD_loss(seg,mask,num_classes)
        loss_prev = loss_prev.cpu().detach().numpy()
        if num_et < thresh:
            seg[:,1,:,:,:] = seg[:,1,:,:,:] + seg[:,0,:,:,:]
            seg[:,0,:,:,:] = seg[:,0,:,:,:]*0
        
        loss_after = MCD_loss(seg,mask,num_classes)
        loss_after = loss_after.cpu().detach().item()
        
        diff_loss = loss_after - loss_prev
        thresh = thresh - lam*(diff_loss)
        print(loss_prev)
        print(loss_after)
        print(thresh)
        torch.cuda.empty_cache()






