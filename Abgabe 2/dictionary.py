import os
import sys


class Dictionary:
    # Brain Storming:
    # -------------------------------------------
    # dictionary aus erlernter Paare
    # wörter nicht unbedingt richtig
    #
    # könnten durch splitted text Dateien iterieren und zur Dict hinufügen
    #

    def __init__(self):
        self.obj_dict = {}  # creates dict

    # updater function
    def update(self, vocab):
        self.obj_dict[vocab] = len(self.obj_dict)
