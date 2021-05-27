import tensorflow as tf
import numpy as np
from tensorflow._api.v2 import data
from tensorflow.python import training
import utility as ut
import os
from tensorflow import keras
from tensorflow.python.keras.layers.core import Dense
import batches
from batches import Batch, get_next_batch
from tensorflow.keras.layers import Input, Concatenate, Embedding
from tensorflow.keras.models import Model

# globals sit here.
from dictionary import dic_src, dic_tar
from utility import cur_dir
from tensorflow.python.keras.backend import _LOCAL_DEVICES


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
    def build_model(self, w=2):
        """
        build our neural network model
        """
        integrate_gpu()

        in_src = []
        out_src = []
        for i in range(2 * w + 1):
            in_src.append(Input(shape=(len(dic_src),), name="I"+str(i)))
            out_src.append(
                Embedding(len(dic_src), 10, input_length=2*w+1)(in_src[i])
            )
        fully_con_src = Concatenate()(out_src)

        # output of fully connected layer
        out_dense_src = Dense(100, activation="relu")(fully_con_src)

        in_tar = []
        out_tar = []

        for i in range(w):
            in_tar.append(Input(shape=(len(dic_tar),), name = "I"+str(2 * w + 1 +i)))
            out_tar.append(Embedding(len(dic_tar), 10, input_length=w)(in_tar[i]))

        fully_con_tar = Concatenate()(out_tar)
        # output of fully connected layer
        out_dense_tar = Dense(100, activation="relu")(fully_con_tar)

        # concatenate output from src and tar in concat layer
        dense_concat = Concatenate(axis = 1)([out_dense_src, out_dense_tar])

        # fully connected layer 1
        fully_connected_one = Dense(100, activation="relu")(dense_concat)

        # second fully connected layer / projection
        fully_connected_two = Dense(100, activation=None, input_shape = (len(dic_tar),))(fully_connected_one)

        # softmax layer
        softmax_layer = Dense(100,activation = "softmax",name = "O")(fully_connected_two)

        # in1 = in_src.extend(in_tar)
        # build final model
        self.model = Model(inputs = [in_src + in_tar] , outputs=[softmax_layer])

    def compile_model(self):
        """
        compiles model
        """
        self.model.compile(
            optimizer="SGD",
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )

        # model.fit()

    def show_summary(self):
        print(self.model.summary())


def feeder(batch):
    pass


def run_nn(sor_file, tar_file, window=2):
    batch = Batch()
    # BUG: REQ das Label muss während das lernen immer bekannt sein. S9 Architektur in letzte VL

    # store source and target file as list of words
    src = ut.read_from_file(sor_file)
    trg = ut.read_from_file(tar_file)

    source, target = batches.get_word_index(src, trg)
    batch, unfinished = Batch(), Batch()
   
    train_model = Feedforward()
    train_model.build_model(window)
    train_model.show_summary()
    train_model.compile_model()

    # running model on the fly
    for s, t in zip(source, target):
        # DONE FIXME: error due to batch < 200

        batch, unfinished = get_next_batch(batch, s, t, window)

        # get next lines to assure batch size exceeds 200 lines
        if batch.size < 200:
            continue

        # create one hot vectors for batch
        # src_window = tf.one_hot(batch.source, depth=len(dic_src))
        # tar_window = tf.one_hot(batch.target, depth=len(dic_tar))
        # labels = tf.one_hot(batch.label, depth=len(dic_tar))

        # use dataset to combine src tar (and labels - later?)
        # for now data.Dataset
        # creates tensors from lists
        feed_src = tf.one_hot(np.array(batch.source),depth = len(dic_src))
        feed_tar = tf.one_hot(np.array(batch.target),depth = len(dic_tar))

        # feed_src = feed_src.map(lambda x: tf.one_hot(x, depth=len(dic_src)))
        # feed_tar = feed_tar.map(lambda x: tf.one_hot(x, depth=len(dic_tar)))
        # feed_zip = tf.data.Dataset.zip((feed_src,feed_tar))
        output_tar = tf.one_hot(np.array(batch.label),depth = len(dic_tar)+len(dic_src))

        # input_src = np.array(feed_src)
        # input_tar = np.array(feed_tar)
        input_list = {}


        for i in range(2*window+1):
            input_list["I"+str(i)] = feed_src[:,i]
        
        for i in range(window):
            input_list["I"+str(i+2*window+1)] = feed_tar[:,i]

        history = train_model.model.fit(input_list, output_tar, batch_size = 1, epochs = 1)
        print(history.history)
        # for step, (src, tar, out) in enumerate(zip(feed_src, feed_tar, output_tar)):
        #     input_list = [np.array(src)]
        #     for element in tar:
        #         input_list.append(np.array(element))
        #     train_model.model.fit([input_list],[out])
        # # uncomment following 2 lines to see the tensor shape
        # running the model

        # run nn

        # set batch to unfinished and work on next batch
        batch = unfinished
        break


def integrate_gpu():
    if not _LOCAL_DEVICES:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def main():
    ## Run neural network
    run_nn(
        os.path.join(cur_dir, "data_exercise_3", "multi30k.en"),
        os.path.join(cur_dir, "data_exercise_3", "multi30k.de"),
    )


main()