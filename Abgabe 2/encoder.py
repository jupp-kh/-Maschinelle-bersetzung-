import dictionary
import sys, time, threading
import utility
import os

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
def get_word_tab(file, n):
    # op_sqnce = []  # sequence of operations

    # list of lines in file
    lis_lines = utility.read_from_file(file)
    word_tab = get_words(lis_lines)  # table of words in file

    for i in range(n):
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
        # op_sqnce.append(max_pair)

        # merge new values to word table
        word_tab = merge_sqnce(word_tab, max_pair)

    print(count_kn_word(word_tab))
    # word_tab.toString()
    return word_tab


#
def subword_split(file, n):
    """performs subword split method on the given file"""
    word_tab = get_word_tab(file, n)  # the op_sqnce is redundant data (ー_ー)!!
    # word_tab.toString()
    # reader; full string from file
    reader = "\n".join(utility.read_from_file(file))

    with open(str(file) + str(n), "w", encoding="utf-8") as write_f:
        # iterate over words in words table
        # if word \neg exists in file then add @@ between pairs
        for word, val in word_tab.tabular.items():
            tmp = word.replace("</w>", "")
            reader = reader.replace(tmp.replace(" ", ""), tmp.replace(" ", "@@ "))

        # write to new file
        write_f.write(reader)


def revert_bpe(file):
    """undos the transformation done to file by BPE"""
    # join all lines into one string separated by \n
    reader = "\n".join(utility.read_from_file(file))

    # write original text back out > file
    with open(file, "w", encoding="utf-8") as write_f:
        write_f.write(reader.replace("@@ ", ""))


def main():
    op_number = [15000]
    for n in op_number:  # replace list with op_number
        subword_split(os.path.join(utility.cur_dir ,"data_exercise_2/multi30k.de"), n)

    # revert_bpe("Abgabe 2/data_exercise_2/multi30k.de100")


if __name__ == "__main__":
    main()
