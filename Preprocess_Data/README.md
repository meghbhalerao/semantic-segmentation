# What preprocessing steps are done?
1. The instructions on how to use the preprocessing scripts is given in the description under the main folder
2. Here, I will just explain in brief how the pre-processing is done
3. First, every image is cropped to remove the 0-intensity value voxels by cropping using a bounding box 
4. Then every image is padded with zeros equally on either sides of every dimension so as to make each of the dimension divisible by 16 (= 2^4 - since the number of the downsampling layers in our network is 4 hence we want every dimension to be divisible by 16 to prevent any dimension issue while upsampling)
5. After this, every image is normalized (except the ground truth :P) using only values in the non-zero region of the image 
6. The above steps are the preprocessing steps that are taken.
7. Other preprocessing steps could also be taken such as bias field correction, SUSAN denoising, but it must be noted that bias field correction must be done on the raw images (without any kind of preporcessing)