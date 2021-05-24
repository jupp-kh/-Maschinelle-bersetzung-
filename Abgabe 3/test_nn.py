import tensorflow as tf
import numpy as np
import os
from tensorflow import keras
import batches
from batches import Batch, get_next_batch
import utility as ut
from dictionary import dic_src, dic_tar

# globals sit here.


# Steps for feed forward network
# we set batches, say a sample matrix of size 3x3
# add weights for layers.
#   could be done using numpy arrays, because of their additional functionality
# set up an activation function in beforehand
# NOTE: using loop to do the feed forward is slow because loops in py are inefficient.
#   -> use %timeit !?

# Architektur aus der Vorlesung
# input layer: src window - target window
#       f[b(i-w): b(i+w)] -
# feste Größe N=200
# zur Darstellung des Vokabulars wird one hot vector benötigt
# Fully connected source  - fully connected target
# concat layer
# fully connected layer 1
# fully connected layer 2 / Projektion
# Ausgabelayer: softmax layer.


class Feedforward:
    """
    Class for our feed forward network
    """

    def __init__(self):
        self.model = None

    # the model
    def build_model(self, batch, w):
        """
        build our neural network model
        """

        # first declare model
        self.model = keras.Sequential(
            [
                keras.layers.Flatten(input_shape=(w, w)),  # input layer for batch
                keras.layers.Dense(
                    128, activation="relu"
                ),  # fully connected source / target
                keras.layers.Dense(10),
            ]
        )

    def compile_model(self):

        self.model.compile(
            optimizer="SGD",
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        # model.fit()


def feeder(batch):
    pass


def run_nn(sor_file, tar_file, window=2):
    batch = Batch()

    # store source and target file as list of words
    src = ut.read_from_file(sor_file)
    trg = ut.read_from_file(tar_file)

    source, target = batches.get_word_index(src, trg)

    # running model on the fly
    for s, t in zip(source, target):
        batch, unfinished = get_next_batch(batch, s, t, window)

        # create one hot vectors for batch
        src_window = tf.one_hot(batch.source, depth=len(dic_src))
        tar_window = tf.one_hot(batch.target, depth=len(dic_tar))
        labels = tf.one_hot(batch.label, depth=len(dic_tar))

        # run nn
        batch = unfinished
