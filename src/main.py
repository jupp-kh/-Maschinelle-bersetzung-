"""
training:
    python3 main.py --tokenise 400 --training-path multi30k 
    without tokenise to directly use multi30k.de/en
testing:
    python3 main.py --seq-path nmt_data/multi30k_sequence_400.de --test-path multi30k
do everything:
    python3 main.py --tokenise 7000 --training-path  --test-path multi30k
"""

import sys
import time
import threading
import os
import batches
import encoder as enc
import dictionary
import recurrent_nn as rnn
import recurrent_dec as rnn_dec
import utility as util

# TODO python script for running the differnt method.


def tokenise_data_bpe(divs, des, train=True, ind=""):
    """creates tokenised data files from original text
    des: holds a subdirectory for data files do not add .en or .de
    returns the paths to both saved files"""

    if train == True:
        commands["seq-file-" + ind], commands["train-" + ind] = enc.run_bpe(
            os.path.join(os.curdir, des), int(divs)
        )
    else:
        commands["test_tkn" + ind] = enc.subword_split(
            os.path.join(os.curdir, des),
            commands["seq-file-" + ind],
            commands["token"],
        )


def overcome_max_line():
    """important for max_line error"""
    rnn.max_line = (
        batches.get_max_line(
            os.path.join(
                os.curdir,
                commands["train-src"],
            ),
            os.path.join(
                os.curdir,
                commands["train-tar"],
            ),
        )
        + 2
    )


def create_bi_dicts():
    """store src and trg dictionaries"""
    read_de = util.read_from_file(os.path.join(os.curdir, commands["train-src"]))
    read_en = util.read_from_file(os.path.join(os.curdir, commands["train-src"]))

    _, _ = batches.get_word_index(read_de, read_en)

    dictionary.dic_src.store_dictionary("source_dictionary_" + str(commands["token"]))
    dictionary.dic_tar.store_dictionary("target_dictionary_" + str(commands["token"]))


# change these parameters for tokenising and training paths
commands = {
    "token": 40,  # number of bpe operations
    "train-src": "train_data/multi30k.de",  # path to training files
    "train-tar": "train_data/multi30k.en",  # path to training files
    "test-src": "test_data/multi30k.dev.de",  # path to test files
    "test-tar": "test_data/multi30k.dev.en",  # path to test files
    "dic-src": "dictionaries/source_dictionary",  # path to dictionaries
    "dic-tar": "dictionaries/target_dictionary",  # path to dictionaries
    "nmt-path": "nmt-data/",  # the path where operations are stored
}

# method done
def nmt_preprocessing():
    """"""
    # remove old data files
    os.system("rm -r nmt_data/*")

    if "token":
        # number of bpe operations
        try:
            tokenise_data_bpe(commands["token"], commands["train-src"], ind="src")
            tokenise_data_bpe(commands["token"], commands["train-tar"], ind="tar")
            tokenise_data_bpe(None, commands["test-src"], train=False, ind="src")
        except Exception as exc:
            print("{}".format(exc))
            print("Error: add all files")
            exit()

    # updating max_line
    overcome_max_line()

    # saving dictionaries
    create_bi_dicts()


def run_nmt():
    """runs neural machine translation"""
    # run training with parameters
    for param in rnn.INFO.keys():
        rnn.INFO[param] = int(input("Input " + param + ": "))

    print(rnn.INFO)
    en_path = os.path.join(os.curdir, commands["train-tar"])
    de_path = os.path.join(os.curdir, commands["train-src"])


def main():
    # nmt_preprocessing()

    # ask for hyperparameters
    run_nmt()


if __name__ == "__main__":
    main()
