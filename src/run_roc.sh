#!/bin/bash

sn_list=("sex" "race")
ratio_list=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
width_list=(2 4 8 16)

## now loop through the above array
for sn in "${sn_list[@]}"
do
    for i in "${width_list[@]}"
    do
        for j in "${ratio_list[@]}"
        do
            python src/run_roc.py -w $i -r $j -sn $sn
        done
    # or do whatever with individual element of the array
    done
done