import sys
import metrics
import subprocess


class Table:
    """ Table saves the learned words """

    # List of pairs and their frequency
    _pair_freq = {}  # dict

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
    def save_highest_pair(self):
        # get max value - max(iterable object, lambda function)
        # lambda gives the value of each key i.e. the second item in tuple
        # val_list = []
        return max(self._pair_freq.items(), key=lambda x: x[1])

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


def get_words(list_words):
    for word in lis_words:
        word_tab.update_pairs(word)


def get_pairs(file):
    # number of operations needed for BDE
    op_number = [1000, 5000, 15000]

    # list of lines in file and list of words in file
    lis_lines = metrics.read_from_file(file)
    lis_words = " ".join(lis_lines).split()

    word_tab = get_words(lis_words)  # table of words in file
    op_sqnce = []  # sequence of operations
    char_list = []

    # iterate over
    for key, value in word_tab.tabular.items():
        # split word to letters and store in list
        characs = list(key)
        characs[len(characs) - 1] += "</w>"
        char_list.append(characs)

    for i in range(1000):
        tmp_table = Table()  # temp structure

        for characs in char_list:
            for j in range(len(characs) - 1):
                tup = characs[j] + characs[j + 1]
                tmp_table.update_pairs(tup)
                tmp_table.tabular[tup] += (
                    word_tab.tabular[characs[: len(characs) - 4]] - 1
                )

    return word_tab


def main():
    t = get_pairs("data_exercise_2/multi30k.de")
    # print(t.tabular)


if __name__ == "__main__":
    main()