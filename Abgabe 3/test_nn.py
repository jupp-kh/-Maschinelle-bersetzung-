from numpy.core.numeric import full
import tensorflow as tf
import numpy as np
import os
from tensorflow import keras
from tensorflow.python.keras.backend import softmax
from tensorflow.python.keras.layers.core import Dense
import batches
from batches import Batch, get_next_batch
import utility as ut
from dictionary import dic_src, dic_tar
from tensorflow.keras.layers import Input, concatenate, Embedding
from tensorflow.keras.models import Model

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
    def build_model(self, w):
        """
        build our neural network model
        """
        # dic_src = range(100)
        # dic_tar = range(100)

        in_src = []
        out_src = []
        for i in range(2 * w + 1):
            in_src.append(Input(shape=(1, len(dic_src))))
            out_src.append(
                Embedding(len(dic_src), 500, input_length=2 * w + 1)(in_src[i])
            )
        fully_con_src = concatenate(out_src)

        # output of fully connected layer
        out_dense_src = Dense(500, activation="relu")(fully_con_src)

        in_tar = []
        out_tar = []

        for i in range(w):
            in_tar.append(Input(shape=(1, len(dic_tar))))
            out_tar.append(Embedding(len(dic_tar), 500, input_length=w)(in_tar[i]))

        fully_con_tar = concatenate(out_tar)

        # output of fully connected layer
        out_dense_tar = Dense(500, activation="relu")(fully_con_tar)

        # concatenate output from src and tar in concat layer
        dense_concat = concatenate([out_dense_src, out_dense_tar])

        # fully connected layer 1
        fully_connected_one = Dense(500, activation="relu")(dense_concat)

        # second fully connected layer / projection
        fully_connected_two = Dense(500, activation=None)(fully_connected_one)

        # softmax layer
        softmax_layer = keras.layers.Softmax()(fully_connected_two)

        # in1 = in_src.extend(in_tar)
        # build final model
        self.model = Model(inputs=[in_src, in_tar], outputs=[softmax_layer])

    def compile_model(self):

        self.model.compile(
            optimizer="SGD",
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=["accuracy", "perplexity"],
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
