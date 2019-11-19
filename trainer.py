################# ALL THE IMPORTS ###################
from __future__ import print_function, division
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.autograd import Variable
from data import TumorSegmentationDataset
from data_val import TumorSegmentationDataset_val
from schd import *
from new_models import fcn,unet,resunet
import gc
from torchsummary import summary
import nibabel as nib
from losses import *
import sys
# The csv files: train_csv and validation_csv, contain a list of patients for training and validation respectively.
# The first column is polulated by the patient name, and the subsequent columns by the each of the 4 modalities (t1, t2, t1ce, flair, and bias field corrected modalities if you wish)
train_csv = sys.argv[1]
validation_csv = sys.argv[2]
# The train_paramters.cfg file is being read as a pandas dataframe for syetematic access to each of the hyperparameters. 
df = pd.read_csv("train_parameters.cfg", sep=' = ', names=['param_name', 'param_value'],
                         comment='#', skip_blank_lines=True,
                         engine='python').fillna(' ')

# Changing the dataframe into a dictionary so that every hyperparameter can be accessed by it's name and more hyperparameters can be added without us worrying about the indexing 
params = {}
for i in range(df.shape[0]):
    params[df.iloc[i, 0]] = df.iloc[i, 1]
n_classes = int(params['n_classes'])
base_filters = int(params['base_filters'])
n_channels = int(params['n_channels'])
num_epochs = int(params['num_epochs'])
model_path = str(params['model_path'])
batch = int(params['batch'])
learning_rate = int(params['learning_rate'])
gt_which = int(params['which_gt'])
log_t = int(params['log_t'])
log_v = int(params['log_v'])
which_loss = params['which_loss']
opt = str(params['opt'])
save_best = int(params['save_best'])
which_model = params['which_model']

# Defining our model here in accordance to the corresponding parameter mentioned in the configuration file 
if which_model == 'resunet':
    model = resunet(n_channels,n_classes,base_filters)
if which_model == 'unet':
    model = unet(n_channels,n_classes,base_filters)
if which_model == 'fcn':
    model = fcn(n_channels,n_classes,base_filters)
if which_model == 'uinc':
    model = uinc(n_channels,n_classes,base_filters)
# Ths is a trivial bit of code to just find out which fold it is in our 5 fold cross validation - this is done by parsing the command line argument string (i.e. the name of the csv file)
strs = os.path.basename(sys.argv[1])
which_gt = [int(s) for s in list(strs) if s.isdigit()]
# a is the fold number 
a = str(which_gt[0])
################################ PRINTING SOME STUFF WITH REGARDS TO THE TRAINING ######################
torch.cuda.empty_cache()
training_start_time = time.asctime()
startstamp = time.time()
print("\nHostname   :" + str(os.getenv("HOSTNAME")))
print("\nStart Time :" + str(training_start_time))
print("\nStart Stamp:" + str(startstamp))
# After each of the sys.stdout.flush() statements the stderr and stdout files are refreshed and new data is entered into them 
sys.stdout.flush()
############################ SETTING THE TRAINING AND VALIDATION DATALOADER ###################
# Setting the training dataloader object which takes the train_csv and type of ground truth (gt_which - either edema, necrosis, and enhancing tumor OR whole tumor, tumor core and enhancing tumor)
dataset_train = TumorSegmentationDataset(train_csv, gt_which)
# Setting up the train_loader with paramters such as the batch size, whether to shuffle or not, number of workers to generate the batches
# If you have to debug the dataloader script data.py or even the neural network architecture : Set the num_workers = 0
train_loader = DataLoader(dataset_train,batch_size= batch,shuffle=True, num_workers=4)
# Same description as the train_loader for the validation loader
dataset_valid = TumorSegmentationDataset_val(validation_csv, gt_which)
val_loader = DataLoader(dataset_valid, batch_size = 1, shuffle = True, num_workers = 4)

#Print the length of the training data 
print("Training Data : ", len(train_loader.dataset))
sys.stdout.flush()
# Search for an available cuda device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Current Device : ", torch.cuda.current_device())
print("Device Count on Machine : ", torch.cuda.device_count())
print("Device Name : ", torch.cuda.get_device_name(device))
print("Cuda Availibility : ", torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
if device.type == 'cuda':
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1),
          'GB')
    print('Cached: ', round(torch.cuda.memory_cached(0)/1024**3, 1), 'GB')
sys.stdout.flush()
# Pushing our model to cuda 
model.cuda()
##################### SETTING THE OPTIMIZER ########################
# Defining our optimizer here in accordance to the corresponding parameter mentioned in the configuration file 
if opt == 'sgd':
    optimizer = optim.SGD(model.parameters(),
                               lr= learning_rate,
                               momentum = 0.9)
if opt == 'adam':    
    optimizer = optim.Adam(model.parameters(), lr = learning_rate, betas = (0.9,0.999), weight_decay = 0.00005)
######## SETTING UP THE LEARNING RATE SCHEDULER ##################
# The learning rate scheduler that I am using right now is a triangular wave between a maximum of 0.001 and a minimum of 0.000001. The reason why I prefer is that it can avoid getting stuck at local minimas by regularly 
# increasing and decreasing the learning rate. Also, the learning rate is updated after every training example (and not after every epoch)
step_size = 4*batch*len(train_loader.dataset)
clr = cyclical_lr(step_size, min_lr = 0.000001, max_lr = 0.001)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])
print(clr(2*step_size))
################DIFFERENT LOSS FUNCTIONS #########################
sys.stdout.flush()
# This is just a helper function which is used to calculate the WT TC and ET dice scores during the training & validation phase of each epoch
def raw_score(out, target):
    oflat = out.view(-1)
    tflat = target.view(-1)
    intersection = (oflat*tflat).sum()
    return (2*intersection)
# Defining our optimizer here in accordance to the corresponding parameter mentioned in the configuration file 
if which_loss == 'dc':
    loss_fn  = MCD_loss
if which_loss == 'dcce':
    loss_fn  = DCCE
if which_loss == 'ce':
    loss_fn = CE
if which_loss == 'mse':
    loss_fn = MCD_MSE_loss
############## DEFINING THE VARIABLES OF THE LOSS FUNCTIONS #################
# Defining the validation loss variables 
avg_val_loss = 0
total_val_loss = 0

avg_c1_val_loss = 0
total_c1_val_loss = 0

avg_c2_val_loss = 0
total_c2_val_loss = 0

avg_c3_val_loss = 0
total_c3_val_loss = 0

best_val_loss = 2000

# Defining the training loss variables 
avg_tr_loss = 0
total_tr_loss = 0

avg_c1_tr_loss = 0
total_c1_tr_loss = 0

avg_c2_tr_loss = 0
total_c2_tr_loss = 0

avg_c3_tr_loss = 0
total_c3_tr_loss = 0

best_tr_loss = 2000


total_loss = 0
total_dice = 0
best_idx = 0

# Defining some validation loss lists
best_n_val_list = []
val_c1_loss_list = []
val_c2_loss_list = []
val_c3_loss_list = []
val_avg_loss_list = []


################ TRAINING THE MODEL ##############
for ep in range(num_epochs):
    start = time.time()
    # Setting the model to train mode
    model.train
    for batch_idx, (subject) in enumerate(train_loader):
        # Load the subject and its ground truth
        image = subject['image']
        mask = subject['gt']
        # Loading images into the GPU and ignoring the affine (just loading them as matrices)
        image, mask = image.float().cuda(), mask.float().cuda()
        # Variable class is deprecated - parameteters to be given are the tensor, whether it requires grad and the function that created it
        image, mask = Variable(image, requires_grad = True), Variable(mask, requires_grad = True)
        # Making sure that the optimizer has been reset  
        optimizer.zero_grad()
        # Forward Propagation to get the output from the models
        output = model(image.float())
        # Computing the loss    
        loss = loss_fn(output.double(), mask.double(), n_classes)
        # Back Propagation for model to learn
        loss.backward()
        #Updating the weight values
        optimizer.step()
        # Calculating the multiclass dice loss and pushing it to the cpu and only taking its value
        curr_loss = MCD_loss(output.double(), mask.double(), n_classes).cpu().data.item()
        # Caculating the total dice loss
        total_loss+=curr_loss
        # Computing the average loss
        average_loss = total_loss/(batch_idx + 1)
        #Computing the dice score 
        curr_dice = 1 - curr_loss
        #Computing the total dice
        total_dice+= curr_dice
        #Computing the average dice
        average_dice = total_dice/(batch_idx + 1)
        # Updating the learning rate according to the learning rate scheduler as mentioned above 
        scheduler.step()
        #Computing the enhancing tumor dice
        loss_c1 = dice_loss(output[:,0,:,:,:].double(),mask[:,0,:,:,:].double())
        loss_c1 = loss_c1.cpu().data.item()
        total_c1_tr_loss+=loss_c1
        avg_c1_tr_loss = total_c1_tr_loss/(batch_idx + 1)
        #Computing the tumor core dice
        loss_c2 =  (raw_score(output[:,1,:,:,:].double(), mask[:,1,:,:,:].double()) + raw_score(output[:,0,:,:,:].double(), mask[:,0,:,:,:].double()))/(torch.sum(output[:,1,:,:,:].double() + mask[:,1,:,:,:].double()) + torch.sum(output[:,0,:,:,:].double() + mask[:,0,:,:,:].double()))
        loss_c2 = 1 - loss_c2.cpu().data.item()
        total_c2_tr_loss+=loss_c2
        avg_c2_tr_loss = total_c2_tr_loss/(batch_idx + 1)
        #Computing the whole tumor dice 
        loss_c3 =  (raw_score(output[:,1,:,:,:].double(), mask[:,1,:,:,:].double()) + raw_score(output[:,0,:,:,:].double(), mask[:,0,:,:,:].double()) + raw_score(output[:,2,:,:,:].double(), mask[:,2,:,:,:].double()))/(torch.sum(output[:,1,:,:,:].double() + mask[:,1,:,:,:].double()) + torch.sum(output[:,0,:,:,:].double() + mask[:,0,:,:,:].double()) + torch.sum(output[:,2,:,:,:].double() + mask[:,2,:,:,:].double()))
        loss_c3 = 1 - loss_c3.cpu().data.item()
        total_c3_tr_loss+=loss_c3
        avg_c3_tr_loss = total_c3_tr_loss/(batch_idx + 1)
        # Printing some information after a log_t interval of training examples 
        if batch_idx%log_t == 0:
            print("Average dice score = " , average_dice, "Current dice score = ", curr_dice )
            print("The learning rate is :", optimizer.param_groups[0]['lr'])
            print("Average Class 1 Training loss is :", avg_c1_tr_loss)
            print("Average Class 2 Training loss is :", avg_c2_tr_loss)
            print("Average Class 3 Training loss is :", avg_c3_tr_loss)
        torch.cuda.empty_cache()    
    # Calculating the average training loss 
    avg_tr_loss = (avg_c1_tr_loss + avg_c2_tr_loss + avg_c3_tr_loss)/3   

    if avg_tr_loss < best_tr_loss:
        best_tr_idx = ep
        best_tr_loss = avg_tr_loss



    total_dice = 0
    total_loss = 0     
    

    total_tr_loss = 0
    avg_tr_loss = 0

    total_c1_tr_loss = 0
    avg_c1_tr_loss = 0

    total_c2_tr_loss = 0
    avg_c2_tr_loss = 0

    total_c3_tr_loss = 0
    avg_c3_tr_loss = 0
    print("Epoch at which best training loss occurs is:", best_tr_idx)
    # Now we enter the evaluation/validation part of the epoch
    
    model.eval
        
    for batch_idx, (subject) in enumerate(val_loader):
        with torch.no_grad():
            image = subject['image']
            mask = subject['gt']
            image, mask = image.cuda(), mask.cuda()
            output = model(image.float())
            #Computing the enhancing tumor dice
            loss_c1 = dice_loss(output[:,0,:,:,:].double(),mask[:,0,:,:,:].double())
            loss_c1 = loss_c1.cpu().data.item()
            total_c1_val_loss+=loss_c1
            avg_c1_val_loss = total_c1_val_loss/(batch_idx + 1)
            #Computing the tumor core dice
            loss_c2 =  (raw_score(output[:,1,:,:,:].double(), mask[:,1,:,:,:].double()) + raw_score(output[:,0,:,:,:].double(), mask[:,0,:,:,:].double()))/(torch.sum(output[:,1,:,:,:].double() + mask[:,1,:,:,:].double()) + torch.sum(output[:,0,:,:,:].double() + mask[:,0,:,:,:].double()))
            loss_c2 = 1 - loss_c2.cpu().data.item()
            total_c2_val_loss+=loss_c2
            avg_c2_val_loss = total_c2_val_loss/(batch_idx + 1)
            #Computing the whole tumor dice 

            loss_c3 =  (raw_score(output[:,1,:,:,:].double(), mask[:,1,:,:,:].double()) + raw_score(output[:,0,:,:,:].double(), mask[:,0,:,:,:].double()) + raw_score(output[:,2,:,:,:].double(), mask[:,2,:,:,:].double()))/(torch.sum(output[:,1,:,:,:].double() + mask[:,1,:,:,:].double()) + torch.sum(output[:,0,:,:,:].double() + mask[:,0,:,:,:].double()) + torch.sum(output[:,2,:,:,:].double() + mask[:,2,:,:,:].double()))


            loss_c3 = 1 - loss_c3.cpu().data.item()
            total_c3_val_loss+=loss_c3
            avg_c3_val_loss = total_c3_val_loss/(batch_idx + 1)

            avg_val_loss = (avg_c1_val_loss + avg_c2_val_loss + avg_c3_val_loss)/3


        if batch_idx%log_v == 0:
            print("Average Validation Loss is : ", avg_val_loss)
            print("Average Class 1 validation loss is :", avg_c1_val_loss)
            print("Average Class 2 Validation loss is :", avg_c2_val_loss)
            print("Average Class 3 Validation loss is :", avg_c3_val_loss)

    

  
    val_c1_loss_list.append(avg_c1_val_loss)
    val_c2_loss_list.append(avg_c2_val_loss)
    val_c3_loss_list.append(avg_c3_val_loss)

    avg_val_loss = (avg_c1_val_loss + avg_c2_val_loss + avg_c3_val_loss)/3
    val_avg_loss_list.append(avg_val_loss)
    
    torch.save(model, model_path + "mod" + a + str(ep) + ".pt")
    
    if ep > save_best:
        keep_list = np.argsort(np.array(val_avg_loss_list))
        keep_list = keep_list[0:save_best]
        for j in range(ep):
            if j not in keep_list:
                if os.path.isfile(os.path.join(model_path + "mod" + a + str(j) + ".pt")):
                    os.remove(os.path.join(model_path + "mod" + a + str(j) + ".pt"))
                
        
        print("Epochs of top 5 validation losses are :", keep_list)

    total_val_loss = 0
    avg_val_loss = 0
    
    total_c1_val_loss = 0
    avg_c1_val_loss = 0

    total_c2_val_loss = 0
    avg_c2_val_loss = 0
    
    total_c3_val_loss = 0
    avg_c3_val_loss = 0

    stop = time.time()
    print("Time taken for epoch no." + str(ep) + " is ",(stop - start)/60)   

d = {}
for i in range(len(df)):
    key = df.iloc[i]['param_name']
    d[key] = df.iloc[i]['param_value']

d['best_epoch'] = save_list[0]

with open('store_file.yaml', 'w') as file:
    documents = yaml.dump(d, file)

