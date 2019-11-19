FILES="/cbica/comp_space/bhaleram/brats/BraTS_2019_Validation/*"
for file in $FILES
do
    echo $(basename $file)
    mkdir /cbica/home/bhaleram/comp_space/brats/new_scripts/29/gen_seg/stored_outputs_val/$(basename $file)
    echo done
done


