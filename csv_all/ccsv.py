import csv 
import os
import pandas as pd
# This script creates 10 csv scripts as given in the readme description 
# Path to folder where all the training datapoints are located
train_path = "/cbica/home/bhaleram/comp_space/fets/data/PreProcessed_Data/train/"
f1 = open('train.csv','w+')
patient_train_list = os.listdir(train_path)
f1.write('t1,t1_bfc,t2, t1ce, t1ce_bfc, flair, flair_bfc, gt\n')
# This is the loop to make the csv file with the list of all the training examples
for patient in patient_train_list:
    f1 = open("train.csv", 'a')
    with f1:
        writer = csv.writer(f1)
        writer.writerow([train_path + patient + '/' + patient + "_t1.nii.gz", 
                         train_path + patient + '/' + patient + "_t1_bfc.nii.gz",
                         train_path + patient + '/' + patient + "_t2.nii.gz",
                         train_path + patient + '/' + patient + "_t1ce.nii.gz",
                         train_path + patient + '/' + patient + "_t1ce_bfc.nii.gz",
                         train_path + patient + '/' + patient + "_flair.nii.gz",
                         train_path + patient + '/' + patient + "_flair_bfc.nii.gz",
                         train_path + patient + '/' + patient + "_seg.nii.gz"])
#This part of the code splits the entire data into 5 folds randomly and creates 10 csvs as given in the description
#This is valid only for 5 fold cross validation 
df = pd.read_csv("train.csv")
df = df.sample(frac = 1)
shp = df.shape
n = shp[0]
e = n%5
n = n + 5 - e
print(n)
df1 = df.iloc[0:n/5, :]
df2 = df.iloc[n/5:2*n/5, :]
df3 = df.iloc[2*n/5:3*n/5, :]
df4 = df.iloc[3*n/5:4*n/5, :]
df5 = df.iloc[4*n/5:n-(5-e), :]
df_t1 = pd.concat((df2,df3,df4, df5))
df_v1 = df1
df_t2 = pd.concat((df1,df3,df4,df5))
df_v2 = df2
df_t3 = pd.concat((df1,df2,df4,df5))
df_v3 = df3
df_t4 = pd.concat((df1,df2,df3,df5))
df_v4 = df4
df_t5 = pd.concat((df1,df2,df3,df4))
df_v5 = df5
df_t1.to_csv("train_fold1.csv", index = False)
df_t2.to_csv("train_fold2.csv", index = False)
df_t3.to_csv("train_fold3.csv", index = False)
df_t4.to_csv("train_fold4.csv", index = False)
df_t5.to_csv("train_fold5.csv", index = False)
df_v1.to_csv("validation_fold1.csv", index = False)
df_v2.to_csv("validation_fold2.csv", index = False)
df_v3.to_csv("validation_fold3.csv", index = False)
df_v4.to_csv("validation_fold4.csv", index = False)
df_v5.to_csv("validation_fold5.csv", index = False)
