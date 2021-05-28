import tensorflow as tf
import numpy as np
import utility as ut
import os
from tensorflow.python.keras.layers.core import Dense
import batches
from batches import Batch, get_next_batch
from tensorflow.keras.layers import Input, Concatenate, Embedding
from tensorflow.keras.models import Model

# globals sit here.
from custom_model import ExtModel, crossentropy, perplexity
from dictionary import dic_src, dic_tar
from utility import cur_dir
from tensorflow.python.keras.backend import _LOCAL_DEVICES

# loading tensorboard
# %load_ext tensorboard

# NOTE: using loop to do the feed forward is slow because loops in py are inefficient.
#   -> use %timeit !?

######################## Adapted Architecture ########################
# input layer: src window - target window
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
            in_src.append(Input(shape=(len(dic_src),), name="I" + str(i)))
            out_src.append(
                Embedding(len(dic_src), 10, input_length=len(dic_src))(in_src[i])
            )
        fully_con_src = Concatenate()(out_src)

        # output of fully connected layer
        out_dense_src = Dense(100, activation="relu")(fully_con_src)

        in_tar = []
        out_tar = []

        for i in range(w):
            in_tar.append(Input(shape=(len(dic_tar),), name="I" + str(2 * w + 1 + i)))
            out_tar.append(
                Embedding(len(dic_tar), 10, input_length=len(dic_tar))(in_tar[i])
            )

        fully_con_tar = Concatenate()(out_tar)
        # output of fully connected layer
        out_dense_tar = Dense(100, activation="relu", kernel_initializer="he_normal")(
            fully_con_tar
        )

        # concatenate output from src and tar in concat layer
        dense_concat = Concatenate(axis=1)([out_dense_src, out_dense_tar])

        # fully connected layer 1
        fully_connected_one = Dense(100, activation="relu")(dense_concat)

        # second fully connected layer / projection
        fully_connected_two = Dense(
            100, activation=None, input_shape=(len(dic_tar), 1)
        )(fully_connected_one)

        # softmax layer
        softmax_layer = Dense(1, activation="softmax", name="O")(fully_connected_two)

        # in1 = in_src.extend(in_tar)
        # build final model
        self.model = Model(inputs=[in_src + in_tar], outputs=[softmax_layer])

    def compile_model(self):
        """
        compiles model
        """
        from tensorflow.keras.optimizers import RMSprop

        self.model.compile(
            optimizer=RMSprop(learning_rate=0.0001),
            loss="mse",
            # using categorical cross entropy from keras provided one-hot vectors
            metrics=[
                "accuracy",
                perplexity,
                tf.keras.metrics.CategoricalCrossentropy(),
            ],
        )

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

        # TODO NEXT: try use dataset to combine src tar (and labels - result)
        # for now data.Dataset
        # creates tensors from lists
        feed_src = tf.one_hot(np.array(batch.source), depth=len(dic_src))
        feed_tar = tf.one_hot(np.array(batch.target), depth=len(dic_tar))

        # feed_src = feed_src.map(lambda x: tf.one_hot(x, depth=len(dic_src)))
        # feed_tar = feed_tar.map(lambda x: tf.one_hot(x, depth=len(dic_tar)))
        # feed_zip = tf.data.Dataset.zip((feed_src,feed_tar))
        output_tar = tf.one_hot(
            np.array(batch.label), depth=len(dic_tar) + len(dic_src)
        )
        # output_tar = tf.reshape(output_tar, (200, 1, len(dic_tar) + len(dic_src)))
        # input_src = np.array(feed_src)
        # input_tar = np.array(feed_tar)

        # dictionary to specify inputs at each input point in NN
        input_list = {}

        for i in range(2 * window + 1):
            input_list["I" + str(i)] = feed_src[:, i]

        for i in range(window):
            input_list["I" + str(i + 2 * window + 1)] = feed_tar[:, i]

        # run nn training with fit
        # FIXME by using fit we assume our entire dataset is fitted which is incorrect: maybe use fit_generator or train_on_batch
        history = train_model.model.fit(
            x=input_list,
            y=output_tar,
            batch_size=20,
            epochs=5,
        )

        # print the returned metrics from our method
        # TODO metriken wir accuracy und perplexity in regelmäßgen Abständen auszugeben
        print(history.history)

        # set batch to unfinished and work on next batch
        batch = unfinished


def integrate_gpu():
    """
    Method to check whether gpu should remain integrated
    """
    if not _LOCAL_DEVICES:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def main():
    ## Run neural network
    run_nn(
        os.path.join(cur_dir, "data_exercise_3", "multi30k.en"),
        os.path.join(cur_dir, "data_exercise_3", "multi30k.de"),
    )


main()