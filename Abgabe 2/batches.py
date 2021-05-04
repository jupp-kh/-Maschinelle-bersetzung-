import os
import metrics
import math


class Batch:
    def __init__(self):
        self._source = []
        self._label = []
        self._target = []

    @property
    def source(self):
        return self._source

    @property
    def label(self):
        return self._label

    @property
    def target(self):
        return self._target

    # setter for all properties
    def set_p(self, src=[], tar=[], lab=[]):
        if src != []:
            self._source = src
        if tar != []:
            self._target = tar
        if lab != []:
            self.label = lab


def tester(w=2, batch_size=200):
    read_de = metrics.read_from_file("Abgabe 2/data_exercise_2/multi30k.de1000")
    read_en = metrics.read_from_file("Abgabe 2/data_exercise_2/multi30k.en1000")
    test_batch = Batch()

    #
    for tar, src in zip(read_de, read_en):
        tar_l = tar.split()
        src_l = src.split()
        tar_len = len(tar_l)
        src_len = len(src_l)
        for i in range(tar_len + 1):
            b_i = alignment(tar_len, src_len)(i)
            while b_i - w + len(test_batch.source) < 0:
                test_batch.source.append(["<s>"])
            else:
                test_batch.set_p(src=[[sub] for sub in src[b_i - w : b_i + w]])
        print(test_batch.source)


def alignment(domain, r_ange):
    """Returns alignment from target to source"""
    op = lambda x: math.floor(x * r_ange / domain)
    return op


tester()
