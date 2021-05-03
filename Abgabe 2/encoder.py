import sys


class Table:
    """ Table saves the learned words """

    # List of pairs and their frequency
    pair_freq = {}  # dict

    # use to save pairs and their frequency
    def update_pairs(self, symbol_pair):
        if symbol_pair in self.pair_freq:
            self.pair_freq[symbol_pair] += 1
        else:
            self.pair_freq[symbol_pair] = 1

    # use to delete pairs
    def rm_pair(self, symbol_pair):
        self.pair_freq.pop(symbol_pair)

    # save pair
    def save_highest_pair(self):
        # get max value - max(iterable object, lambda function)
        # lambda gives the value of each key i.e. the second item in tuple
        # val_list = []
        return max(self.pair_freq.items(), key=lambda x: x[1])

        # # find all occurences of max_val
        # for key, val in self.pair_freq.items():
        #     if val == max_val[1]:
        #         val_list.append(key)

        # return val_list  # return list of all keys


def caller():
    op_number = [1000, 5000, 15000]


x = Table()
x.update_pairs("j")
x.update_pairs("s")
x.update_pairs("s")
x.update_pairs("s")
x.update_pairs("s")
x.update_pairs("s")
x.update_pairs("r")
x.update_pairs("r")
x.update_pairs("r")
x.update_pairs("r")
x.update_pairs("r")

print(x.pair_freq)
print(x.save_highest_pair())