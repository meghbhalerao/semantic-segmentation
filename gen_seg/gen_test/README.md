# How are the segmentations generated on unknown data?
Again, ideally, the documentation of this entire repo is written under the assumption that the user is working on a SGE based HPC cluster.

**NEEDS FINISHING**: 
- Where are the following variables coming from: `data_path`, `save_path`, `model_path1`.
- How to choose the appropriate model path by looking at the standard output file from the logs.


1. This is the folder you want to `cd` into when you want to generate tumor segmentations from Brain MR Images
2. The segmentation is generated using 5 models (which are trained using 5 fold cross validation), by first individually generating predictions of each model and then combining them using majority voting.
3. In this folder you can see the files named as `seg_single_model*.*` and `submit_single*.sh`.
4. These files are used to generate the segmentations of individual models and the fold number can be obtained from the naming scheme of the files. 
5. Run each of the `seg_single_model*.sh` either by `bash` or `qsub` - doesn't really matter. 
6. Running these bash scripts will submit `submit_single*.sh` script to the cluster multiple times with the patient name as the paramter to the script
7. The entire patient image is segmented at once and not patch-wise (like the training process) and hence the memory requirement is high and hence the inference can't be done on a GPU.
8. Further details of the memory requirements can be found in either of the `submit_single*.sh` scripts.
9. For further details on how and where the predicted segmentations are stored (of each fold) please `cd` into the `stored_outputs_val` folder under `gen_seg`
10. Once you have understood how the 5 segmentations from the 5 folds are generated, now we can fuse all the 5 segmentations using majority voting 
11. Change the `path` variable to the path to the folder `stored_outputs_test` and the `save_path` variable to the folder where you want to store the final segmentations.
12. Once you have made these changes run the `majority_voting.sh` script 
13. Running this script will combine the predictions of the 5 folds using majority voting and save them accoring to the paths mentioned 
