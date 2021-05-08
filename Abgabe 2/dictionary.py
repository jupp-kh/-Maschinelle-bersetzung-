import os
import sys


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
    def update(self, vocab):
        """adds index and word to dictionary"""
        self.bi_dict[vocab] = len(self.obj_dict)

    def get_word(self, index):
        """gets word at index in dictionary"""
        return self.bi_dict.keys()[index]

    def get_index(self, word):
        """gets index of word in dictionary"""
        return self.bi_dict[word]

    def toString(self):
        """prints out dictionary"""
        print(self.bi_dict)