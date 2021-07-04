"""
recurrent_dec.py offers methods to compute beam search and greedy search 
translations are made using the RNN model from recurrent_nn.py
"""
import itertools
import math
import os
import multiprocessing
from numpy.lib.utils import source
import tensorflow as tf
from tensorflow.keras.backend import argmax
from tensorflow.keras.backend import get_value
import numpy as np
from encoder import revert_bpe
from batches import (
    Batch,
    create_batch,  # TODO remove later
    get_word_index,  # TODO remove later
)
import utility as ut
import recurrent_nn as rnn
from dictionary import dic_tar, dic_src
from utility import cur_dir, read_from_file


def get_model():
    """builds and returns a model from model's path"""
    enc, dec = get_enc_dec_paths()
    test_model = rnn.Translator(len(dic_tar), len(dic_src), 200, 200, 200)

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


def save_k_txt(file_txt, k):
    """provided an integer k and encoded text saves beam predictions into file system"""
    keys_list = dic_tar.get_keys()
    txt_list = [[] for _ in range(k)]
    for elem in file_txt:
        sentence = [[] for _ in range(k)]
        for i in range(k):
            str_lines = map(lambda x: keys_list[x], elem[i][0])
            sentence = list(str_lines)
            txt_list[i].append(sentence)
    for i in range(k):
        ut.save_list_as_txt(
            os.path.join(
                cur_dir,
                "en_de_translation",
                "beam_k=" + str(k) + "_prediction" + str(i) + ".de",
            ),
            txt_list[i],
        )
        revert_bpe(
            os.path.join(
                cur_dir,
                "en_de_translation",
                "beam_k=" + str(k) + "_prediction" + str(i) + ".de",
            )
        )


def line_beam(i, source, k):
    """used by beam search to process lines simultaneously"""
    # nonlocal file_txt, test_model
    t_1, t_2 = 0, 0
    # i, (s, t) in enumerate(zip(source, target)):

    batch = source
    candidate_sentences = [[[0], 0.0]]
    pred_values = []

    # loading model
    test_model = get_model()
    for iterator in range(len(batch.source)):
        all_candidates = []
        for j, _ in enumerate(candidate_sentences):
            # get last element to compute next prediction
            t_2 = candidate_sentences[j][0][-1]
            # print(t_2)
            dic = {
                "I0": np.array([batch.source[iterator]]),
                "I1": np.array([[t_1, t_2]]),
            }
            # prediction step
            pred_values = test_model.predict(dic, batch_size=1, callbacks=None)[0]

            # swap values and get the best predictions
            t_1 = t_2
            k_best = tf.math.top_k(pred_values, k=k)
            seq, score = candidate_sentences[j]
            for x in enumerate(k_best):
                candidate = [
                    seq + [get_value(k_best.indices[x])],
                    score - math.log(get_value(k_best.values[x])),
                ]
                all_candidates.append(candidate)
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        candidate_sentences = ordered[:k]
    return candidate_sentences


def beam_decoder(source, k):
    """finds the best translation scores using the beam decoder."""
    file_txt = []

    # open pool for multiprocessing library
    with multiprocessing.Pool(processes=8) as pool:
        # multiprocessing lines
        file_txt = pool.starmap(
            line_beam,
            zip(
                range(10),
                source[:10],
                itertools.repeat(k),
            ),
        )
    # ensure the pipe is closed and wait for all
    # processes to finish their work
    pool.close()
    pool.join()

    # save the predicted outputs
    save_k_txt(file_txt, k)


def rnn_pred_batch(source_file, target_file):
    """returns a batch from source file by padding all sentences"""
    source_file, _ = get_word_index(source_file, target_file)
    source_file = list(map(lambda x: list(reversed(x)), source_file))

    # instead of maxlen=46 use max_word_in_line
    return tf.keras.preprocessing.sequence.pad_sequences(source_file, maxlen=46)


def get_enc_dec_paths():
    """returns encoder and decoder path as tuple"""
    enc_path = os.path.join(cur_dir, "rnn_checkpoints", "encoder.epoch01-loss2.70.hdf5")
    dec_path = os.path.join(cur_dir, "rnn_checkpoints", "decoder.epoch01-loss2.70.hdf5")
    # print(get_model(enc_path, dec_path, 200))
    return (enc_path, dec_path)


def main():
    """main method"""
    rnn.init_dics()
    source = read_from_file(os.path.join(cur_dir, "test_data", "multi30k.dev.de"))
    target = read_from_file(os.path.join(cur_dir, "test_data", "multi30k.dev.en"))

    inputs = rnn_pred_batch(source, target)  # preprocessed batches from source file


def rec_dec_tester():
    """called for testing specific methods"""
    rnn.init_dics()
    m = get_model()


if __name__ == "__main__":
    # main()
    rec_dec_tester()
