# How are the segmentations generated on unknown data?
Again, ideally, the documentation of this entire repo is written under the assumption that the user is working on a SGE based HPC cluster.
1. This is the folder you want to `cd` into when you want to generate tumor segmentations from Brain MR Images
2. The segmentation is generated using 5 models (which are trained using 5 fold cross validation), by first individually generating predictions of each model and then combining them using majority voting.
3. In this folder you can see the files named as `seg_single_model*.*` and `submit_single*.sh`.
4. These files are used to generate the segmentations of individual models and the fold number can be obtained from the naming scheme of the files. 
5. 