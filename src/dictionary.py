"""
contains Dictionary class, which includes a bidirectional dictionary 
for quick mapping between words and integer values.
"""
import os
import utility as ut
from utility import cur_dir


class Dictionary:
    """# Brain Storming:
    # -------------------------------------------
    # dictionary aus erlernter Paare
    # wörter nicht unbedingt richtig?
    #
    # könnten durch splitted text Dateien iterieren und zur Dict hinufügen
    """

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

    def __len__(self):
        return len(self.bi_dict)

    def store_dictionary(self, file_name):
        """ Stores dictionary pairs into specified file file_name """
        ut.save_list_as_txt(
            os.path.join(cur_dir, "dictionaries", file_name), self.bi_dict.items()
        )

    def get_stored(self, file_name):
        """ Used to retrieve stored dictionary from previous sessions """
        dic_list = ut.read_from_file(os.path.join(cur_dir, "dictionaries", file_name))
        for word in dic_list:
            self.update(word.split()[0])

    def translate_to_nums(self, words):
        """
        uses bi_dict to replace strings with their index
        """
        pass

    def translate_to_words(self, numbers):
        """
        uses bi_dict to replace numbers with their key
        """
        pass


###  Globals
dic_tar = Dictionary()
dic_src = Dictionary()