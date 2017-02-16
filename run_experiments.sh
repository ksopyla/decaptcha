#!/bin/bash

learning_rates=(0.001 0.002 0.005 0.01)
dropout=(0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9)

iters=40000

SECONDS=0

for l in "${learning_rates[@]}"
do
	for d in "${dropout[@]}"
    do 
        python decaptcha_convnet.py -l $l -d $d -i $iters
    done
done

ELAPSED="Elapsed: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"
echo $ELAPSED


