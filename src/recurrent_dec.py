"""
recurrent_dec.py offers methods to compute beam search and greedy search 
translations are made using the RNN model from recurrent_nn.py
"""
import os
import math
import time
import multiprocessing
import itertools
import tensorflow as tf
import numpy as np
import random

from tensorflow.keras import metrics
import batches
import metrics
from encoder import revert_bpe
import utility as ut
import recurrent_nn as rnn
from dictionary import dic_tar, dic_src
from utility import cur_dir, read_from_file
import tensorflow_addons as tfa
from tensorflow.keras.backend import argmax
from tensorflow.keras.backend import get_value

# TODO create pool to translate the hole dev file

# +2 for start and end of sequence symbol
max_line = (
    batches.get_max_line(
        os.path.join(cur_dir, "train_data", "multi30k_subword.de"),
        os.path.join(cur_dir, "train_data", "multi30k_subword.en"),
    )
    + 2
)


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
                "de_en_translation",
                "beam_k=" + str(k) + "_prediction" + str(i) + ".de",
            ),
            txt_list[i],
        )
        revert_bpe(
            os.path.join(
                cur_dir,
                "de_en_translation",
                "beam_k=" + str(k) + "_prediction" + str(i) + ".de",
            )
        )


def roll_out_encoder(sentence, search=True, batch_size=1):
    """builds and returns a model from model's path"""
    enc, dec = get_enc_dec_paths()
    test_model = rnn.Translator(len(dic_tar), len(dic_src), 200, 200, batch_size)
    dec_input = [[dic_src.bi_dict["<s>"]] + [0 for _ in range(max_line - 1)]]
    dec_input = np.array(dec_input)
    temp = tf.zeros((batch_size, dec_input.shape[1]), dtype=tf.float32)

    enc_output, _ = test_model.encoder(temp, None)

    dec_output, weights, state = test_model.decoder((temp, enc_output), None)

    test_model.encoder.load_weights(enc)
    test_model.decoder.load_weights(dec)
    if search:
        sentence = np.array([sentence])
        enc_output, _ = test_model.encoder(sentence, None)
        dec_output, weights, state = test_model.decoder((dec_input, enc_output), None)

    return test_model, enc_output, dec_output


def translate_sentence(sentence, k=1, one_line=False):
    """translates sentence using beam search algorithm"""

    model, enc_output, dec_output = roll_out_encoder(sentence)
    first_pred = tf.math.top_k(dec_output, k)
    # TODO cut the search when end sequence appears in different sentence
    candidate_sentences = []
    for i in range(k):
        candidate_sentences.append(
            [
                [dic_tar.bi_dict["<s>"], get_value(first_pred.indices[0][0][i])],
                -math.log(get_value(first_pred.values[0][0][i])),
            ]
        )
    pred_values = []
    for index in range(1, max_line):
        all_candidates = []
        for j, _ in enumerate(candidate_sentences):
            pre_pred_word = candidate_sentences[j][0]
            if pre_pred_word[-1] == dic_tar.bi_dict["</s>"]:
                all_candidates.append(candidate_sentences[j])
                continue

            pre_sentence = tf.keras.preprocessing.sequence.pad_sequences(
                [pre_pred_word], maxlen=max_line, value=0, padding="post"
            )
            pred_word, _, _ = model.decoder((pre_sentence, enc_output))

            k_best = tf.math.top_k(pred_word, k=k)

            seq, score = candidate_sentences[j]
            for x, _ in enumerate(k_best.indices):
                candidate = [
                    seq + [get_value(k_best.indices[0][index][x])],
                    score - math.log(get_value(k_best.values[0][index][x])),
                ]
                all_candidates.append(candidate)
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        candidate_sentences = ordered[:k]

    if one_line:
        for sen in candidate_sentences:
            print_sentence(sen[0])

    return candidate_sentences


# TODO translate with bigger batch_size dont forger remainder sentences
def beam_decoder(source, k, save=False):
    """finds the best translation scores using the beam decoder."""
    file_txt = []
    set_off = time.time()
    for src in source:
        file_txt.append(translate_sentence(src, k))

    print("Time taken fo {}: {} sec".format(k, time.time() - set_off))
    # TODO add the blue metrik and print it.

    # save the predicted outputs
    if save:
        save_k_txt(file_txt, k)

    return file_txt


def rnn_pred_batch(source_list):
    """returns a batch from source file by padding all sentences"""
    for i, _ in enumerate(source_list):
        tmp = [
            dic_src.get_index(x)
            if x in dic_src.bi_dict
            else random.randint(0, len(dic_src.bi_dict))
            for x in source_list[i].split(" ")
        ]
        source_list[i] = tmp
    source_list = list(
        map(
            lambda x: [dic_src.bi_dict["<s>"]] + list(x) + [dic_src.bi_dict["</s>"]],
            source_list,
        )
    )
    # instead of maxlen=46 use max_word_in_line
    return tf.keras.preprocessing.sequence.pad_sequences(
        source_list, maxlen=max_line, value=0, padding="post"
    )


def print_sentence(pred):
    res = []
    keys = dic_tar.get_keys()
    for x in pred:
        res.append(keys[x])
    print(" ".join(res))


def get_enc_dec_paths():
    """returns encoder and decoder path as tuple"""
    enc_path = os.path.join(cur_dir, "rnn_checkpoints", "encoder.epoch27-loss0.46.hdf5")
    dec_path = os.path.join(cur_dir, "rnn_checkpoints", "decoder.epoch27-loss0.46.hdf5")

    return (enc_path, dec_path)


# TODO from terminal with runing with differnt model
# FIXME i am not effizien
def bleu_score(source, target, k=1, n=4):
    source = rnn_pred_batch(source)
    pred = beam_decoder(source, k)
    keys_list = dic_tar.get_keys()
    # list of texts
    txt_list = []
    for elem in pred:
        # list of line string
        sentences = []
        for i in range(k):
            str_lines = map(lambda x: keys_list[x], elem[i][0])
            sentence = list(str_lines)
            sentence = " ".join(sentence)
            sentence = sentence.replace("<s> ", "")
            sentence = sentence.replace("</s>", "")
            sentence = sentence.replace("@@ ", "")
            sentences.append(sentence)
        txt_list.append(sentences)

        for txt in txt_list:
            print(txt)
            print(metrics.met_bleu(txt, target, n, False))


def main():
    """main method"""
    rnn.init_dics()
    source = read_from_file(
        os.path.join(cur_dir, "test_data", "multi30k.dev_subword.de")
    )
    target = read_from_file(os.path.join(cur_dir, "test_data", "multi30k_mini.en"))

    inputs = [
        "eine person in bluejeans und rotem pullover biegt um die ecke einer ziegelmauer . ",
        "eine person in bluejeans und rotem pullover biegt um die ecke einer ziegelmauer . ",
        "eine person in bluejeans und rotem pullover biegt um die ecke einer ziegelmauer . ",
    ]
    x = rnn_pred_batch(
        [
            "eine person in bluejeans und rotem pullover biegt um die ecke einer ziegelmauer . "
        ]
    )[0]
    # inputs = rnn_pred_batch(source)
    # translate_sentence(x, 1, True)
    # beam_decoder(inputs, 1, True)
    bleu_score(inputs, target, 3)

    # inputs = tf.convert_to_tensor(inputs)
    # print(inputs)
    # f, s = translate_line(1, inputs, 1)
    # print(f, s)


def rec_dec_tester():
    """called for testing specific methods"""
    rnn.init_dics()
    inputs = rnn_pred_batch(["ein mann schläft in einem grünen raum auf einem sofa ."])
    target = rnn_pred_batch(["a man sleeping in a green room on a couch ."])

    m = roll_out_encoder(None)


if __name__ == "__main__":
    main()
    # rec_dec_tester()
