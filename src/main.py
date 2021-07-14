"""
training:
    python3 main.py --tokenise 400 --training-path multi30k 
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


def tokenise_data_bpe(divs, des, train=True):
    """creates tokenised data files from original text
    des: holds a subdirectory for data files do not add .en or .de
    returns the paths to both saved files"""
    for i in [".en", ".de"]:
        if train == True:
            commands["seq-file" + i], commands["train_tkn_file" + i] = enc.run_bpe(
                os.path.join(os.curdir, des + i), int(divs)
            )
        else:
            commands["test_tkn_file" + i] = enc.subword_split(
                os.path.join(os.curdir, des + i), commands["seq-file" + i]
            )


def launch_dictionary_update(src, trg, bpe):
    """launches and creates the bidirectional dictionary for src and trg"""
    path1 = os.path.join(os.curdir, "dictionaries", "source_dic_" + bpe)
    path2 = os.path.join(os.curdir, "dictionaries", "target_dic_" + bpe)

    util.save_list_as_txt(path1, src, strings=True)
    util.save_list_as_txt(path2, trg, strings=True)
    return path1, path2


def overcome_max_line():
    """important for max_line error"""
    rnn.max_line = (
        batches.get_max_line(
            os.path.join(
                os.curdir,
                commands["--train-data"] + "_subword_" + commands["--tokenise"] + ".de",
            ),
            os.path.join(
                os.curdir,
                commands["--train-data"] + "_subword_" + commands["--tokenise"] + ".en",
            ),
        )
        + 2
    )
    rnn_dec.max_line = rnn.max_line


def create_bi_dicts():
    """store src and trg dictionaries"""
    read_de = util.read_from_file(
        os.path.join(
            os.curdir,
            commands["--train-data"] + "_subword_" + commands["--tokenise"] + ".de",
        )
    )
    read_en = util.read_from_file(
        os.path.join(
            os.curdir,
            commands["--train-data"] + "_subword_" + commands["--tokenise"] + ".en",
        )
    )

    #
    _, _ = batches.get_word_index(read_de, read_en)
    dictionary.dic_src.store_dictionary("source_dictionary_" + commands["--tokenise"])
    dictionary.dic_tar.store_dictionary("target_dictionary_" + commands["--tokenise"])


commands = {
    "--tokenise": None,  # number of bpe operations
    "--training-path": "train_data/",  # path to training files
    "--test-path": "test_data/",  # path to test files
    "--dic-path": "dictionaries/",  # path to dictionaries
    "--seq-path": "nmt-data/",  # the path where operations are stored
    "--train-data": "nmt_data/",  # the exact file for training tokenised
    "--test-data": "nmt_data/",  # the exact file for testing tokenised
}


def main():
    args = sys.argv
    if "--training-path" in args:
        commands["--training-path"] += args[args.index("--training-path") + 1]
        commands["--train-data"] += args[args.index("--training-path") + 1]

    if "--seq-path" in args:
        commands["--seq-path"] += args[args.index("--seq-path") + 1]

    if "--test-path" in args:
        commands["--test-path"] += args[args.index("--test-path") + 1]
        commands["--test-data"] += args[args.index("--test-path") + 1]

    if "--dic-path" in args:
        commands["--dic-path"] += args[args.index("--dic-path") + 1]

    # commands ready and good to go
    # preprocessing data
    if "--tokenise" in args:
        commands["--tokenise"] = args[args.index("--tokenise") + 1]

        # get training data ready
        if "--training-path" in args:
            # saves the training data in nmt
            tokenise_data_bpe(
                commands["--tokenise"],
                commands["--training-path"],
                train=True,
            )

        # get test data ready
        if "--test-path" in args:
            tokenise_data_bpe(None, commands["--test-path"], train=False)

    # overcome error from max_line in rnn
    overcome_max_line()  # call on nmt/"tokenised data file"

    create_bi_dicts()
    print(len(dictionary.dic_tar.bi_dict))
    print(len(dictionary.dic_src.bi_dict))
    # create dictionary files for training data
    # initialise both dictionaries


if __name__ == "__main__":
    main()
