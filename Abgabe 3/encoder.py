import dictionary
import sys, time, threading
import utility
import os
import ntpath

# Globals
# testing dictionary
# the data structure needs some work
# deutsch_dict = dictionary.Dictionary()

# based on table in page 23 in "Folien zum Vokabular und Subwords Units"


class Table:
    """ Table saves the learned words """

    # List of pairs and their frequency
    def __init__(self):
        self._pair_freq = {}  # dict

    @property
    def tabular(self):
        return self._pair_freq

    # use to save pairs and their frequency
    def update_pairs(self, symbol_pair):
        if symbol_pair in self._pair_freq:
            self._pair_freq[symbol_pair] += 1
        else:
            self._pair_freq[symbol_pair] = 1

    # use to delete pairs
    def rm_pair(self, symbol_pair):
        self._pair_freq.pop(symbol_pair)

    # returns highest pair
    def get_highest_pair(self):
        # get max value - max(iterable object, lambda function)
        # lambda gives the value of each key i.e. the second item in tuple
        # val_list = []
        return max(self._pair_freq.items(), key=lambda x: x[1])[0]
        # # find all occurences of max_val
        # for key, val in self._pair_freq.items():
        #     if val == max_val[1]:
        #         val_list.append(key)

        # return val_list  # return list of all keys

    # TODO maybe get rid of this or better output form ;]
    def toString(self):
        """Prints Table object as a table with columns pair and frequency"""
        for key, value in self.tabular.items():
            print("(", key, value, ")", end="\t")
        print("")


# contruct a table of words from the list of passed words
def get_words(lis_lines):
    """returns Table of words from the list of passed words"""
    word_tab = Table()
    lis_words = " ".join(lis_lines).split()
    for word in lis_words:
        # the Table structure updates itself by incrementing the number of x
        # occurences for some word w
        # word format: the word "lesen" becomes "l e s e n</w>" etc
        word_tab.update_pairs(" ".join(list(word)) + "</w>")  # t h e</w>
    return word_tab


def count_kn_word(word_tab):
    """ count number of known words """
    kn_counter = 0
    sw_counter = 0
    for key, val in word_tab.tabular.items():
        if len(key.split()) == 1:
            kn_counter += 1
        sw_counter += len(key.split())
        # TODO: maybe remove this, as it is just an assumption for
        #       that our dictionary would store the tokens/subwords.
        # save learned subwords in german dictionary
        #     deutsch_dict.update(key[0:-4])
        # else:
        #     for sub in key[0:-4].split():
        #         deutsch_dict.update(sub)
    return kn_counter, sw_counter


#
def merge_sqnce(word_tab, max_pair):
    """Used to merge the maximum pair in file"""
    tmp_table = Table()
    for key, value in word_tab.tabular.items():
        # max pair is passed splitted
        # remove space and
        hold_key = key.replace(max_pair, max_pair.replace(" ", ""))
        tmp_table.tabular[hold_key] = value

    return tmp_table


# FIXME change name, as op_sqnce is redundant
def get_op_sequences(file, n):
    op_squences = []  # sequence of operations

    # list of lines in file
    lis_lines = utility.read_from_file(file)
    word_tab = get_words(lis_lines)  # table of words in file

    for _ in range(n):
        tmp_table = Table()  # temp structure

        # go through keys
        # count all occurrences of symbol pairs
        for key, value in word_tab.tabular.items():
            sym_list = key.split()
            for j in range(len(sym_list) - 1):
                pair = sym_list[j] + " " + sym_list[j + 1]
                tmp_table.update_pairs(pair)
                tmp_table.tabular[pair] += value - 1

        # get maximum pair
        if not tmp_table:
            return word_tab
        max_pair = tmp_table.get_highest_pair()

        # add max to operation sequence
        op_squences.append(str(max_pair))

        # merge new values to word table
        word_tab = merge_sqnce(word_tab, max_pair)

    # word_tab.toString()
    return op_squences


def create_op_sequences(file, n):
    op_sequences = get_op_sequences(file, n)
    file_des = os.path.join(
        utility.cur_dir,
        "output",
        ntpath.splitext(ntpath.basename(file))[0]
        + "_op_sequence_"
        + str(n)
        + ntpath.splitext(ntpath.basename(file))[1]
        + ".csv",
    )
    utility.save_as_csv(file_des, op_sequences)


#


def subword_split(text_file, sequence_file):
    """
    runs subword split on text_file
    """
    lines_list = utility.read_from_file(text_file)
    text = ""

    #
    for line in lines_list:
        for word in line.split():
            text += " ".join(list(word)) + "</w> "
        text += "\n"

    # read operation sequence from file
    op_sequences = utility.read_from_file(sequence_file)

    for sequence in op_sequences:
        text = text.replace(sequence, sequence.replace(" ", ""))

    text = text.replace(" ", "@@ ").replace("</w>@@", "")
    file_des = os.path.join(
        utility.cur_dir,
        "output",
        ntpath.splitext(ntpath.basename(text_file))[0]
        + "_subword"
        + ntpath.splitext(ntpath.basename(text_file))[1],
    )

    utility.save_as_txt(file_des, text)


def revert_bpe(file):
    """undos the transformation done to file by BPE"""
    # join all lines into one string separated by \n
    reader = "\n".join(utility.read_from_file(file))

    # write original text back out > file
    with open(file, "w", encoding="utf-8") as write_f:
        reader = reader.replace("<s> ", "")
        reader = reader.replace("</s>", "")
        write_f.write(reader.replace("@@ ", ""))


def run_bpe(*oper):
    for n in oper:  # replace list with op_number
        # create_op_sequences(
        #    os.path.join(utility.cur_dir, "data_exercise_3/multi30k.de"), n
        # )
        # create_op_sequences(
        #    os.path.join(utility.cur_dir, "data_exercise_3/multi30k.en"), n
        # )

        subword_split(
            os.path.join(utility.cur_dir, "data_exercise_3/multi30k.dev.de"),
            os.path.join(
                utility.cur_dir, "output", "multi30k_op_sequence_" + str(n) + ".de.csv"
            ),
        )
        subword_split(
            os.path.join(utility.cur_dir, "data_exercise_3", "multi30k.dev.en"),
            os.path.join(
                utility.cur_dir, "output", "multi30k_op_sequence_" + str(n) + ".en.csv"
            ),
        )
    # revert_bpe("Abgabe 2/data_exercise_2/multi30k.de100")


def rename_me():
    for i in range(5):
        revert_bpe(
            os.path.join(
                os.curdir, "predictions", "beam_k=10_prediction" + str(i) + ".de"
            )
        )
    # run_bpe(7000)


if __name__ == "__main__":
    rename_me()