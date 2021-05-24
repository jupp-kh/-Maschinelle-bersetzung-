"""
contains Dictionary class, which includes a bidirectional dictionary 
for quick mapping between words and integer values.
"""


class Dictionary:
    # Brain Storming:
    # -------------------------------------------
    # dictionary aus erlernter Paare
    # wörter nicht unbedingt richtig?
    #
    # könnten durch splitted text Dateien iterieren und zur Dict hinufügen
    #

    def __init__(self):
        self.bi_dict = {}  # creates dict

    # updater function
    def update(self, *vocabs):
        """adds index and word to dictionary"""
        for vocab in vocabs:
            if vocab not in self.bi_dict:
                self.bi_dict[vocab] = len(self.bi_dict)

    def get_word(self, index):
        """gets word at index in dictionary"""
        return list(self.bi_dict.keys())[index]

    def get_index(self, word):
        """gets index of word in dictionary"""
        return self.bi_dict[word]

    def get_keys(self):
        return list(self.bi_dict.keys())

    def __str__(self):
        """prints out dictionary"""
        return str(self.bi_dict)


###  Globals
dic_tar = Dictionary()
dic_src = Dictionary()