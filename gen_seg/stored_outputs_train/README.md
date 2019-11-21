# What does this file structure mean?
1. All the folders are the patient names of the train dataset.
2. Inside each of the patient folders is the predicted segmentation of each of the fold which is generated from the stuff that we do in the `gen_train` 
3. Every patient folder will have (provided that the stuff in `gen_train` ran error-free) 5 `.nii.gz` (nifti) files which will be names this way: `$patient_nameX.nii.gz` where `X=1:5` depending on the fold, corresponding to the predictions of each fold.
4. These 5 predictions will be combined using majority voting. For details on how to do the majority voting please navigate to the `gen_train` folder.   
