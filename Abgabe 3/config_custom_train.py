# ---- config file for feedforward neural network training ----

# This file contains paths and parameters for a custom training of a feed forward neural network
# Start the training with the parameters by running the 'run_me.py' script (and set flags)

paths = dict(
    ## following two blocks contain files ##
    ### currently: targen: en, source:de ###
    # text file to train the model on 
    # should exist in data_exercise_3
    source_train_file="multi30k_subword.en",
    target_train_file="multi30k_subword.de",
    # text file to validate trained model on
    # should also exist in data_exercise_3  
    source_val_file="multi30k.dev_subword.en",
    target_val_file="multi30k.dev_subword.de",
)
params = dict(
    # Print reports frequency
    # prints training metrics every 50 batch as default
    reports="50",
    # checkpoint saving frequency
    cp_freq="1000",
    # number of operations for subword split : BPE algorithm
    # default is set to 7000 operations
    oper="7000",
    ## Option to cut the learning rate in halve 
    # set to true when the performance of the model is stagnant 
    fractional_lr="False",

    ## print tensorboard
    tb="True",
)
