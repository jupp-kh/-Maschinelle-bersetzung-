from json import encoder
from traceback import print_tb
import numpy as np
from tensorflow.core.protobuf.config_pb2 import OptimizerOptions
from tensorflow.python.keras.backend import gradients, variable
from test_nn import outdated_tester
from metrics import met_bleu
from keras.layers import Dense
import tensorflow as tf
from dictionary import dic_tar, dic_src
import os
from utility import cur_dir
import sys
import decoder
import batches
import glob

# from decoder import loader, greedy_decoder

from tensorflow.python.ops.variables import trainable_variables

# import config file for hyperparameter search space
import config_custom_train as config


class Encoder(tf.keras.Model):
    def __init__(self, dic_size, em_dim, num_units, batch_size, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.num_units = num_units
        self.embedding = tf.keras.layers.Embedding(dic_size, em_dim, name="Embedding")
        self.lstm = tf.keras.layers.LSTM(
            num_units,
            activation="sigmoid",
            return_state=True,
            return_sequences=True,
            name="LSTM",
        )

    def call(self, em, hidden):
        """implements call from keras.Model"""
        # specify embedding input and pass in embedding in lstm layer
        em = self.embedding(em)
        print(em)
        output, h, c = self.lstm(em, initial_state=hidden)
        return output, h, c

    def initialize_hidden_state(self):
        return [
            tf.zeros((self.batch_size, self.num_units)),
            tf.zeros((self.batch_size, self.num_units)),
        ]


class Decoder(tf.keras.Model):
    def __init__(self, dic_size, em_dim, num_units, batch_size, **kwargs):
        """implements init from keras.Model"""
        super(Decoder, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.num_units = num_units

        # input layer and lstm layer
        self.embedding = tf.keras.layers.Embedding(dic_size, em_dim, name="Embedding")
        self.lstm = tf.keras.layers.LSTM(
            num_units,
            activation="sigmoid",
            return_state=True,
            return_sequences=True,
            name="LSTM",
        )

        self.flatten = tf.keras.layers.Flatten(name="Flatten")
        # output layer
        self.softmax = Dense(dic_size, activation="softmax", name="Softmax")

    def call(self, inputs, enc_output):
        """implements call from keras.Model"""
        em = self.embedding(inputs)
        mask = self.embedding.compute_mask(inputs)
        output, _, _ = self.lstm(em, initial_state=enc_output, mask=mask)
        flat = self.flatten(output)
        return self.softmax(flat)


# init dictionaries
def init_dics():
    """read learned dictionaries for source and target"""
    dic_src.get_stored(os.path.join(cur_dir, "dictionaries", "source_dictionary"))
    dic_tar.get_stored(os.path.join(cur_dir, "dictionaries", "target_dictionary"))


def categorical_loss(real, pred):
    """computes and returns categorical cross entropy"""
    entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(
        real, pred
    )
    # mask unnecessary symbols
    mask = tf.logical_not(tf.math.equal(real, 0))
    mask = tf.cast(mask, dtype=entropy.dtype)
    entropy *= mask

    # compute the mean of elements across dimensions of entropy
    return tf.reduce_mean(entropy)


@tf.function
def train_model(input, targ, hidden):
    """method for RNN train step"""
    loss = 0

    with tf.GradientTape() as tape:
        # pass input into encoder
        output, h, c = encoder(input, hidden)

        # TODO later attention mechanism

        # pass input into decoder
        dec_output = decoder(targ, [h, c])
        loss = categorical_loss(targ, dec_output)

    var = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, var)

    # apply gradients and return loss
    optimizer.apply_gradients(zip(gradients, var))
    return loss


def main():
    """main method"""
    init_dics()
    encoder = Encoder(len(dic_src), 200, 200, 200)
    decoder = Decoder(len(dic_tar), 200, 200, 200)

    en_path = os.path.join(cur_dir, "train_data", "min_train.en")
    de_path = os.path.join(cur_dir, "train_data", "min_train.de")
    batch = batches.create_batch_rnn(de_path, en_path)

    global optimizer
    optimizer = tf.keras.optimizers.Adam()
    # dataset = tf.data.Dataset(np.array(batch.source))(
    #     batch_size=200, drop_remainder=True
    # )
    o, h, c = encoder(np.array(batch.source[:200]), encoder.initialize_hidden_state())
    print(encoder.summary())

    o = decoder(np.array(batch.target[:200]), [h, c])
    print(decoder.summary())


if __name__ == "__main__":
    main()
