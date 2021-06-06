#!/usr/bin/bash

########## parameters ##########

# Print reports frequency
# prints training metrics every 50 batch as default
reports="50"
# checkpoint saving frequency
cp_freq="30"

# number of operations for subword split : BPE algorithm
# default is set to 7000 operations
oper="7000"

## following two blocks contain files ##
### currently: targen: en, source:de ###
# text file to train the model on 
# should exist in data_exercise_3
source_train_file="multi30k_subword.en"
target_train_file="multi30k_subword.de"

# text file to validate trained model on
# should also exist in data_exercise_3  
source_val_file="multi30k.dev_subword.en"
target_val_file="multi30k.dev_subword.de"

## saving and loading models
# checkpoints can also be specified 
# TODO code here

## Option to cut the learning rate in halve 
# set to true when the performance of the model is stagnant 
fractional_lr="False"

## print tensorboard
tb="True"




#######    Now we call test_nn with the specified parameters     #######
if [ ! -z $1 ]
then 
    if [ $1 = "--go" ] 
    then 
        args="${reports} ${oper} ${source_train_file} ${target_train_file} ${source_val_file} ${target_val_file} ${cp_freq} ${fractional_lr} ${tb}"

        python3 test_nn.py ${args} 
        exit
    else 
        echo "Error " $1 " is not a valid argument"
    fi
fi


echo "This script runs test_nn with specified parameters"
echo "Open the file and change the values "
echo "When done, run the script as such: ./run_me.bash --go"
