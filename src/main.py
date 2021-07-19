"""
automated steps
"""

import utility as util
import os
import batches
import encoder as enc
import dictionary
import recurrent_nn as rnn
import recurrent_dec as rnn_dec
import utility as util
import matplotlib.pyplot as plt


def tokenise_data_bpe(divs, des, train=True, ind=""):
    """creates tokenised data files from original text
    des: holds a subdirectory for data files do not add .en or .de
    returns the paths to both saved files"""

    if train == True:
        commands["seq-file-" + ind], commands["train-" + ind] = enc.run_bpe(
            os.path.join(util.cur_dir, des), int(divs)
        )
    else:
        commands["test_tkn_" + ind] = enc.subword_split(
            os.path.join(util.cur_dir, des),
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
    rnn_dec.max_line = (
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
    read_en = util.read_from_file(os.path.join(os.curdir, commands["train-tar"]))

    _, _ = batches.get_word_index(read_de, read_en)

    dictionary.dic_src.store_dictionary("source_dictionary_" + str(commands["token"]))
    dictionary.dic_tar.store_dictionary("target_dictionary_" + str(commands["token"]))


# change these parameters for tokenising and training paths
commands = {
    "token": None,  # number of bpe operations
    "train-src": "train_data/multi30k.de",  # path to training files
    "train-tar": "train_data/multi30k.en",  # path to training files
    "test-src": "test_data/multi30k.dev.de",  # path to test files
    "test-tar": "test_data/multi30k.dev.en",  # path to test files
    "dic-src": "dictionaries/source_dictionary",  # path to dictionaries
    "dic-tar": "dictionaries/target_dictionary",  # path to dictionaries
    "nmt-path": "nmt_data/",  # the path where operations are stored
}

# method done
def nmt_preprocessing():
    """"""
    # remove old data files
    # os.system("rm -r nmt_data/*")
    if not os.path.exists(os.path.join(util.cur_dir, commands["nmt-path"])):
        os.makedirs(os.path.join(util.cur_dir, commands["nmt-path"]))
    if commands["token"]:
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
        try:
            rnn.INFO[param] = int(input("Input " + param + ": "))
        except Exception as exc:
            print("Inputs should be integers!")
            print("Exiting ... ")
            exit()

    print("------------------------------------------------------")
    print("Training parameters: ")
    print(rnn.INFO)
    print("------------------------------------------------------")
    en_path = os.path.join(os.curdir, commands["train-tar"])
    de_path = os.path.join(os.curdir, commands["train-src"])
    data = rnn.preprocess_data(en_path, de_path)[1]
    rnn.train_loop(
        rnn.INFO["EPOCHS"],
        data,
        rnn.INFO["BATCH_SZ"],
        rnn.INFO["MET_RATE"],
        rnn.INFO["CP_RATE"],
        rnn.INFO["CP_START"],
    )


def decode_results():
    """runs beam search"""
    cp_path = rnn.INFO["CP_DIR"]

    files_enc = []
    files_dec = []
    train_loss = {}
    dev_bleu = {}

    for file in os.listdir(cp_path):
        if file.endswith(".hdf5"):
            if file.startswith("encoder"):
                files_enc.append(file)
            if file.startswith("decoder"):
                files_dec.append(file)

    cp_dir_name = str(cp_path.split("/")[-1])

    for e, d in zip(sorted(files_enc), sorted(files_dec)):
        if not e[8:] == d[8:]:
            print(
                f"Some CP file seems to be missing! Check directory '{cp_dir_name}' for missing files!"
            )
            exit()
        act_epoch = int(e.split("epoch")[1].split("-")[0])
        train_loss[act_epoch] = float(e.split("loss")[1].split(".hdf5")[0])

        rnn.init_dics()
        source = util.read_from_file(
            commands["test_tkn_src"]
        )  # path to tokenized de file
        commands["test-tar"] = os.path.join(util.cur_dir, commands["test-tar"])
        target = util.read_from_file(commands["test-tar"])  # path to multi30k.dev.en

        enc_path = os.path.join(cp_path, e)
        res, bleu = rnn_dec.bleu_score(source, target, 200, path=enc_path)
        rnn_dec.save_translation(res, bleu)
        print("BLEU for Epoch", int(e.split("epoch")[1].split("-")[0]), "is:", bleu)
        dev_bleu[act_epoch] = bleu

    x = sorted(train_loss.keys())
    y_train_loss = [train_loss[a] for a in x]

    y_dev_bleu = [dev_bleu[a] for a in x]

    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.plot(x, y_train_loss, "x-", color="b", label="train loss")

    ax2 = ax.twinx()
    ax2.plot(x, y_dev_bleu, "x-", color="r", label="dev BLeu")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax2.set_ylabel("BLEU")

    img_path = os.path.join(os.curdir, "plots", cp_dir_name + "_fig1.png")
    print(os.path.join(os.curdir, "plots", "fig1.png"))
    # make sure directory exists
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    plt.savefig(img_path)


def main():
    commands["token"] = input("Input BPE Merge Operations: ")
    #
    nmt_preprocessing()

    print("------------------------------------------------------")
    print("List of working directories. \n", commands)
    print("------------------------------------------------------")

    # ask for hyperparameters
    # run training
    rnn.init_dics("_" + commands["token"])
    run_nmt()

    # start the decoding process
    # beam search from recurrent dec
    decode_results()


if __name__ == "__main__":
    main()
