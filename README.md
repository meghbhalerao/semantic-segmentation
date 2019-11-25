# Medical Image Semantic Segmentation

This repository gives the step-by-step instructions to train different variations of the U-Net architecture for Medical Image Semantic Segmentation (using PyTorch). As of now this process is limited to Brain Tumor Segmentation, but will later be extended to other anatomies.  
Clone this repository using `git clone` and then `cd` into it. Ideally, this documentation is written under the assumption that the user is working on a SGE based HPC cluster (Hence, the terminology that I will be using subsequently will be influenced by this assumption) with sufficient GPU and CPU memory. The CPU and GPU memory requirements are further elaborated in the documentation in the respective folders. 


## What are the modules/packages needed?
PyTorch framework is used. For uses on the `cbica-cluster` (although you can see the required modules at the beginning of each trainer script I will mention them here anyway):

```bash
module load pytorch/1.0.1 # Always needed
module load gcc/5.2.0 # Needed somtimes because pandas package throws errors sometimes if this is not loaded 
```

## Preparing the dataset 

### Downloading the dataset
Set of instructions on how to preprocess the raw BraTS data:
1. Follow instructions on [this page](https://www.med.upenn.edu/cbica/brats2019/registration.html) to download the BraTS training data
2. Create a new folder called `Original_data` and within it a folder called `train`:
```bash
cd ${brats_data} # the data was downloaded and extracted in this location
mkdir Original_data
cd Original_data
mkdir train
```
3. Copy/Move all the patients from the `HGG` and `LGG` folders into the folder `train` which is mentioned above (wherever it may be located)
```bash
cd ${brats_data}
mv HGG/* Original_data/train/
mv LGG/* Original_data/train/
```

### Folder Structure of the Dataset
1. All the scripts (whichever are relavent) are written with repect to the data folder structure of the BraTS dataset.
2. So, it is important to note that, if one is not using the BraTS data and/or is using different/additional data, it must comply with the BraTS dataset folder structure which will be described in the subsequent points.
3. Let's take the case of the data in `Original_data/train/` with `n` patients.
4. `n` patients correspond to `n` folders in the `Original_data/train/`
5. The name of each of these folders is the `patient_ID` of that particular patient.
6. The `patient_ID` could be any alpha-numeric sequence.
7. Each `patient_ID` folder consists 5 `*.nii.gz` files with the following names : `patient_ID_t1.nii.gz`, `patient_ID_t2.nii.gz`, `patient_ID_t1ce.nii.gz`, `patient_ID_flair.nii.gz` and `patient_ID_seg.nii.gz`, corresponding to 4 imaging modalities and 1 ground truth segmentation mask.

So, in short, whatever data you use, it is expected to be in the folder structure that is explained in the points above.

### Preprocessing the dataset (for more details look into the `Preprocess_Data` folder)
1. Open `${repo_location}/Preprocess_Data/pp.py` with your favorite editor and change the variable `path_data` to `${brats_data}/Orignal_data/train/` as mentioned in the point number 2 above (Do not forget the `/` at the end - this is assumed to be present during file parsing).
2. Change the `path` variable in `${repo_location}/Preprocess_Data/pp.py` to the folder where you wish to save the preprocessed data, preferably to something understandable such as `${brats_data}/Preprocessed_data/train/` (this location needs to be present **before** the script runs).
3. Run the file `pp.py` using `python ${repo_location}/Preprocess_Data/pp.py` after making sure that all the dependencies [numpy, math, nibabel, tqdm] are installed. Doing this will preprocess the raw data and write it to the location specified in `path`.

###  Preparing the necessary CSV files (for more details look into the `csv_all` folder)
1. Open the `${repo_location}/csv_all/ccsv.py` with your favorite editor and change the variable `train_path` to the path defined by the `path` variable in the previous section, i.e., `${brats_data}/Preprocessed_data/train/`.
2. Run the `ccsv.py` using `python ${repo_location}/csv_all/ccsv.py`, again after making sure that the necessary dependencies [csv, pandas] are installed.
3. The training process is done using `5` fold cross validation, hence `10` CSV files are generated in the location defined by `train_path`: `5` each for training and validation folds.

## Training the Model

### Setting the training hyperparameters
Open the `${repo_location}/train_parameters.cfg` file and change the training hyperparameters such as Number of Epochs (`num_epochs`), Optimizer (`opt`), Loss Function (`which_loss`), batch size and so on. The descriptions for each of the hyperparameters is documented in the file itself.

**TODO : ADD VARIABLE FOR PAUSE-RESUME TRAINING** 

###  Running the training process (for more details look into the `submission_scripts` folder)
1. The training script `${repo_location}/submission_scripts/trainer.py` takes in 2 command line arguments : first is the path to the training `csv` file of a given fold, and the second is the path to the validation `csv` file of a given fold (the respective pairs are generated at point **4** of the CSV preparation step above).
2. `cd` into the `${repo_location}/submission_scripts` folder. There are 5 submission scripts (one for each fold).
3. Edit each of the submission scripts to make sure that the correct paths to the training and validation scripts is passed as arguments to `trainer.py` (these are generated in the CSV file section).
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
