FILES="/cbica/home/bhaleram/comp_space/fets/data/PreProcessed_Data/brats_test/*"
for file in $FILES
do
    echo $(basename $file)
    mkdir /cbica/home/bhaleram/comp_space/fets/new_scripts/FCN/Exp_1/gen_seg/stored_outputs_test/$(basename $file)
    echo done
done


