# Instructions on how to use these scripts to train the models
1. Each of the training scripts is for one particular training and validation fold. 
2. Open either one of the files to have a look at how the submission script is written
3. The header of each of the files gives us the description on the requirements and other details (It can be easily understood after opening either of the files)
4. Also, please note you must provide the training and validation csv paths as 2 arguments to the trainer.py - can be understood after looking into the `trainer_f*.sh` script
5. To run the training process submit the jobs one by one using `qsub $script_name.sh` for each of the training scripts
6. 5 `stderr` and 5 `stdout` files (2 for each fold) will be created with their names according to the `$job_name` specified in the submission scripts where the outputs of the trainer will be logged 
