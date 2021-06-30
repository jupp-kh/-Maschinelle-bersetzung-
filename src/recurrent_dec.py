"""
recurrent_dec.py offers methods to compute beam search and greedy search 
translations are made using the RNN model from recurrent_nn.py
"""
import itertools
import math
import os
import multiprocessing
import tensorflow as tf
from tensorflow.keras.backend import argmax
from tensorflow.keras.backend import get_value
import numpy as np
from encoder import revert_bpe
from batches import (
    Batch,
    get_pred_batch,
    create_batch,  # TODO remove later
    get_word_index,  # TODO remove later
)
import utility as ut
import recurrent_nn as rnn
from dictionary import dic_tar, dic_src
from utility import cur_dir


def get_model(enc, dec, batch_size):
    """builds and returns a model from model's path"""
    test_model = rnn.Translator(len(dic_tar), len(dic_src), 200, 200, batch_size)

    # initialize layers
    _, h, c = test_model.encoder(
        tf.zeros((200, 46)), test_model.encoder.initialize_hidden_state()
    )
    test_model.decoder(tf.zeros((200, 46)), [h, c])

    test_model.encoder.load_weights(enc)
    test_model.decoder.load_weights(dec)

    # compile then return loaded model
    test_model.compile(
        optimizer="adam",
    )
    return test_model


rnn.init_dics()
enc_path = os.path.join(cur_dir, "rnn_checkpoints", "encoder.epoch01-loss2.70.hdf5")
dec_path = os.path.join(cur_dir, "rnn_checkpoints", "decoder.epoch01-loss2.70.hdf5")
print(get_model(enc_path, dec_path, 200))