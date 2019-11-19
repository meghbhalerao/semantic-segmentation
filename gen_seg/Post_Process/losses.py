import numpy as np
import torch 
def dice_loss(inp, target):
    smooth = 1e-7
    iflat = inp.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))



def MCD_loss(pm, gt, num_class):
    acc_dice_loss = 0
    for i in range(0,num_class):
        acc_dice_loss += dice_loss(gt[:,i,:,:,:],pm[:,i,:,:,:])
    acc_dice_loss/= num_class
    return acc_dice_loss

# Setting up the Evaluation Metric
def dice(out, target):
    smooth = 1e-7
    oflat = out.view(-1)
    tflat = target.view(-1)
    intersection = (oflat * tflat).sum()
    return (2*intersection+smooth)/(oflat.sum()+tflat.sum()+smooth)


def CE(out,target):
    oflat = out.view(-1)
    tflat = target.view(-1)
    loss = torch.dot(-torch.log(oflat), tflat)/(oflat.sum()+tflat.sum())
    return loss

def DCCE(out,target, n_classes):
    l = MCD_loss(out,target, n_classes) + CE(out,target)
    return l

