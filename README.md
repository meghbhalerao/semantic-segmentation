# Medical Image Semantic Segmentation
This repository gives the step-by-step instructions to train different variations of the U-Net architecture for Medical Image Semantic Segmentation (using PyTorch). As of now this process is limited to Brain Tumor Segmentation, but will later be extended to other anatomies.  
Clone this repository using `git clone` and then `cd` into it. Ideally, this documentation is written under the assumption that the user is working on a SGE based HPC cluster (Hence, the terminology that I will be using subsequently will be influenced by this assumption) with sufficient GPU and CPU memory. The CPU and GPU memory requirements are further elaborated in the documentation in the respective folders. 

## What are the modules/packages needed?
PyTorch framework is used. For uses on the `cbica-cluster` (although you can see the required modules at the beginning of each trainer script I will mention them here anyways):
`module load pytorch/1.0.1` : Always needed
`module load gcc/5.2.0` : Needed somtimes because pandas package throws errors sometimes if this is not loaded 

## Preparing the dataset 

### Downloading the dataset
Set of instructions on how to preprocess the raw BraTS data:
1. Download the BraTS training data from here : https://www.google.com/search?client=ubuntu&channel=fs&q=cbica+ipp&ie=utf-8&oe=utf-8
2. Create a new folder called `Original_data` and within it a folder called `train`
3. Copy/Move all the patients from the `HGG` and `LGG` folders into the folder `train` which is mentioned above (wherever it may be located)
### Folder Structure of the Dataset
1. All the scripts (whichever are relavent) are written with repect to the data folder structure of the BraTS dataset. 
### Preprocessing the dataset (for more details look into the `Preprocess_Data` folder)
1. `cd` into the `Preprocess_Data` folder and open the `pp.py`  and change the variable `path_data` to `/$Whatever/Orignal_data/train/` as mentioned in the point number 2 above (Do not forget the `/` at the end - this is assumed to be present during file parsing)
2. Change the `path` variable to the folder where you wish to save the preprocessed data, preferably to something understandable such as `/$Whatever/Preprocessed_data/train/` (this location needs to be present **before** the script runs)
3. Run the file `pp.py` using `python pp.py` after making sure that all the dependencies [numpy, math, nibabel, tqdm] are installed. Doing this will preprocess the raw data and write it to the location specified in `path`.

###  Preparing the necessary CSV files (for more details look into the `csv_all` folder)
1. `cd` into the `csv_all` folder
2. Open the `ccsv.py` file and change the variable `train_path` to the path defined by the `path` variable in the previous section, i.e., `/$Whatever/Preprocessed_data/train/`.
3. Run the `ccsv.py` using `python ccsv.py`, again after making sure that the necessary dependencies [csv, pandas] are installed
4. The training process is done using 5 fold cross validation, hence 10 `.csv` files are generated, 5 for training folds and 5 for validation folds

## Training the Model

### Setting the training hyperparameters
Open the `train_parameters.cfg` file and change the training hyperparameters such as Number of Epochs (`num_epochs`), Optimizer (`opt`), Loss Function (`which_loss`), batch size and so on. The description of each of the hyperparameter is documented in the [train_parameters.cfg](https://github.com/meghbhalerao/Semantic_Segmentation/blob/master/train_parameters.cfg) file

###  Running the training process (for more details look into the `submission_scripts` folder)
1. The training script `trainer.py` takes in 2 command line arguments : first is the path to the training `csv` file of a given fold, and the second is the path to the validation `csv` file of a given fold (the respective pairs are generated at point **4** of the CSV preparation step above).
2. `cd` into the `submission_scripts` folder. There are 5 submission scripts (one for each fold)
3. Edit each of the submission scripts to make sure that the correct paths to the training and validation scripts is passed as arguments to `trainer.py`
4. Run each of the submission scripts (`trainer_f*.sh`) either by `bash script_name.sh` or `qsub script_name.sh` (if you are using a SGE computing cluster)

###  How and where are the weights saved?
1. The weights (models) are saved as `*.pt` files.
2. All the models will be saved in the folder that you specified in the `model_path` parameter in the `train_parameters.cfg` file
3. The saved models follow is specific naming scheme
4. Each of the models is named in the form `modXYYY.pt`
5. `X` tells us which fold is the model from i.e. `1-5` and `YYY` tells us what is the epoch number where the best model weights were obtained.
6. Depending on the value of the parameter `save_best` in the `train_paramters.cfg` file, `$save_best` number of models will be saved. Hence, the total number of weight files that will be saved are : `$save_best * number_of_folds`

## Inference/Predicting Segmentation on unseen data (for more details look into the `gen_seg` folder)
`cd` into the `gen_seg` folder which in-short stands for **generate segmentations**. After this you can `cd` into either of `gen_train` or `gen_validation` or `gen_test` according to which dataset's segmentation you want to generate.
