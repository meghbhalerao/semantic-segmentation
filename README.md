# Medical Image Semantic Segmentation
Variations of the U-Net architecture for Medical Image Semantic Segmentation. Clone this repository using `git clone` and then `cd` into it. 
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
1. 
