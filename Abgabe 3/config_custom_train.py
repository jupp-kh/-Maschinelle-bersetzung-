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
    cp_freq="100000",
    # number of operations for subword split : BPE algorithm
    # default is set to 7000 operations
    oper="7000",
    ## Option to cut the learning rate in halve 
    # set to true when the performance of the model is stagnant 
    fractional_lr="True",

    ## print tensorboard
    tb="True",
)
# search space for hyperparameter search
hp_space = dict(
    # windowsize
    w=2,
    # embedding output dimension source
    output_dim_emb = dict(
        name = 'emb_output_dim',
        min_value=200,
        max_value=1000,
        step=100,
    ),
    units_fullcon = dict(
        name = 'units_fullcon',
        min_value=100,
        max_value=1000,
        step=100,
    ),
    units_fullcon_1 = dict(
        name = 'units_fullcon_1',
        min_value=100,
        max_value=1000,
        step=100,
    ),
    units_fullcon_2 = dict(
        name = 'units_fullcon_1',
        min_value=100,
        max_value=1000,
        step=100,
    ),
    activation_functions = dict(
        name='activation_functions',
        functions=['relu','sigmoid','tanh'],
    ),
    learning_rates = dict(
        name='learning_rates',
        lrs=[1e-2, 1e-3, 1e-4],
    ),
)
search_params = dict(
    epochs=1,                       # something between 10-15 maybe
    max_trials=1,                   # trials for models with random parameters each time -> depends on time/resources you have :D
    executions_per_trial=1,         # how often the model with same params should be trained fully (can be different each time)
    directory_name='hp_search-01',
)
