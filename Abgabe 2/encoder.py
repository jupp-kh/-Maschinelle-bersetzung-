import sys
import metrics
import subprocess
import time


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

    def toString(self):
        """Prints Table object as a table with columns pair and frequency"""
        print("pair     |     freq.")
        for elem in self._pair_freq.items():
            print(elem[0], "       |    ", elem[1])


#
def get_words(lis_lines):
    word_tab = Table()
    lis_words = " ".join(lis_lines).split()
    for word in lis_words:
        word_tab.update_pairs(" ".join(list(word)) + "</w>")
    return word_tab


#
def merge_sqnce(word_tab, max_pair):
    """Used to merge the maximum pair in file"""
    tmp_table = Table()
    for key, value in word_tab.tabular.items():
        hold_key = key.replace(max_pair, max_pair.replace(" ", ""))
        # print(hold_key)
        # time.sleep(2)
        tmp_table.tabular[hold_key] = value

    return tmp_table


def get_pairs(file):
    # number of operations needed for BDE
    op_number = [1000, 5000, 15000]
    op_sqnce = []  # sequence of operations

    # list of lines in file
    lis_lines = metrics.read_from_file(file)
    word_tab = get_words(lis_lines)  # table of words in file

    for i in range(1000):
        tmp_table = Table()  # temp structure

        if i % 100 == 0:
            print(op_sqnce)
        # go through keys
        # count all occurrences of symbol pairs
        for key, value in word_tab.tabular.items():
            sym_list = key.split()
            for j in range(len(sym_list) - 1):
                pair = sym_list[j] + " " + sym_list[j + 1]
                tmp_table.update_pairs(pair)
                tmp_table.tabular[pair] += value - 1

        # get maximum pair
        max_pair = tmp_table.get_highest_pair()

        # add max to operation sequence
        op_sqnce.append(max_pair)

        # merge new values to word table
        word_tab = merge_sqnce(word_tab, max_pair)

    return word_tab


def main():
    t = get_pairs("Abgabe 2/data_exercise_2/multi30k.de")
    # print(t.tabular)


if __name__ == "__main__":
    main()