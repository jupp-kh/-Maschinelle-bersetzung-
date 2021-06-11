import config_custom_train as config
import sys
import os

###############################################
#
#   This script runs the different training methods for our feed forward model
#       1.  -   Custom Training
#       2.  -   Hyperparameter Search & Safe best model with Keras Tuner
#
#   Flag explanations
#       --go        :
#       --hypparam  :
#       --gpu       :
#
###############################################

# Read arguments and save bools in dict to decide which how to run the training
arguments = {
    'go': False,
    'hypparamsearch': False,
    'gpu': False,
}

args = sys.argv[1:]
for arg in args:
    if arg == '--go':
        arguments['go'] = True
    elif arg == '--hypparam':
        arguments['hypparamsearch'] = True
    elif arg == '--gpu':
        arguments['gpu'] = True
    else:
        raise ValueError('Invalid command line argument/flag! Please use valid set of flags/arguments!')

if not arguments['go']:
    print("Error: Please enter argument '--go' to start the training process!")
    sys.exit()

# Get parameters from config file
test_nn_args=f"{config.params['reports']} {config.params['oper']} {config.paths['source_train_file']} "\
            f"{config.paths['target_train_file']} {config.paths['source_val_file']} {config.paths['target_val_file']} "\
            f"{config.params['cp_freq']} {config.params['fractional_lr']} {config.params['tb']}"

# Generate string with bool from flag existance
flags = f" {arguments['hypparamsearch']} {arguments['gpu']}"

# run script with parameters
os.system("python3 test_nn.py "+test_nn_args)
