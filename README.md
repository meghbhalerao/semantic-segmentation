# Medical Image Semantic Segmentation
Variations of the U-Net architecture for Medical Image Semantic Segmentation using PyTorch. Clone this repository using `git clone` and then `cd` into it. 
## Preparing the dataset 
### Downloading the dataset
Set of instructions on how to preprocess the raw BraTS data:
1. Download the BraTS training data from here : https://www.google.com/search?client=ubuntu&channel=fs&q=cbica+ipp&ie=utf-8&oe=utf-8 
2. Create a new folder called `Original_data` and within it a folder called `train`
3. Copy/Move all the patients from the `HGG` and `LGG` folders into the folder `train` which is mentioned above (wherever it may be located)
### Preprocessing the dataset
1. `cd` into the `Preprocess_data` folder and open the `pp.py`  and change the variable `path_data` to `/$Whatever/Orignal_data/train/` as mentioned in the point number 2 above (Do not forget the `/` at the end)
2. Change the `path` variable to the folder where you wish to save the preprocessed data, preferably to something understandable such as `/$Whatever/Preprocessed_data/train/`
3. Run the file `pp.py` using `python pp.py` after making sure that all the dependencies are installed. Doing this will preprocess the raw data
###  Preparing the necessary CSV files
1. `cd` into the `csv_all` folder
2. Open the `ccsv.py` file and change the variable `train_path` to  `/$Whatever/Preprocessed_data/train/` as mentioned above.
3. Run the `ccsv.py` using `python ccsv.py`, again after making sure that the necessary dependencies are installed
4. The training process is done using 5 fold cross validation, hence 10 `.csv` files are generated, 5 for training folds and 5 for validation folds
## Training the Model
### Setting the training hyperparameters
Open the `train_parameters.cfg` file and change the training hyperparameters such as Number of Epochs, Optimizer, Loss Function, etc. The description of each of the hyperparameter is documented in the `train_parameters.cfg` file
###  Running the training process
1. The training script `trainer.py` takes in 2 command line arguments : first is the path to the training `csv` file of a given fold, and the second is the path to the validation `csv` file of a given fold.
2. `cd` into the `submission_scripts` folder. There are 5 submission scripts (one for each fold)
3. Edit each of the submission scripts to make sure that the correct paths to the training and validation scripts is passed as an argument to `trainer.py` 
###  How and where are the weights saved?


