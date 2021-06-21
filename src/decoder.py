"""
not empty
"""
import itertools
from pstats import SortKey
from metrics import compare_bleu_scores

from encoder import revert_bpe
import math
import os
from tensorflow.keras.backend import argmax
from tensorflow.keras.backend import get_value
import tensorflow as tf
import numpy as np
from batches import *
import utility as ut
import custom_model as cm
from dictionary import dic_tar, dic_src
from utility import cur_dir
import multiprocessing


def get_model(model):
    """builds and returns a model from model's path"""
    test_model = cm.WordLabelerModel()
    # print(len(dic_src), len(dic_tar))
    # print(test_model.summary())
    test_model.load_weights(
        model,
    )
    test_model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        # using categorical cross entropy from keras provided one-hot vectors
        metrics=[
            "accuracy",
            cm.Perplexity(),
        ],
    )
    return test_model


def create_text_files(line, k):
    """Used with beam decoder to store the created lines from predictions. """
    sentence = [[] for _ in range(k)]
    keys_list = dic_tar.get_keys()
    for i in range(k):
        str_lines = map(lambda x: keys_list[x], line[i][0])
        sentence = " ".join(str_lines)
        ut.save_line_as_txt(
            os.path.join(cur_dir, "predictions", "beam_prediction" + str(i) + ".de"),
            sentence,
        )


def greedy_decoder(test_model, source, target):
    """searches through the output of predict using greedy search"""
    history = []
    greedy_values = []
    t_1, t_2 = 0, 0

    #
    for i, (s, t) in enumerate(zip(source, target)):
        batch = create_batch(Batch(), s, t)

        pred_values, tmp_greed = [], []
        for iterator in range(len(batch.source)):
            dic = {
                "I0": np.array([batch.source[iterator]]),
                "I1": np.array([[t_1, t_2]]),
            }
            # prediction step
            pred_values.append(test_model.predict(dic, batch_size=1, callbacks=None))

            # swap values and get the best predictions
            t_1 = t_2
            t_2 = get_value(argmax(pred_values[iterator]))[0]
            tmp_greed.append(t_2)

        greedy_values.append(tmp_greed)
        history.append(pred_values)
    print(history)

    dic_keys = dic_tar.get_keys()
    path = os.path.join(cur_dir, "predictions", "greedy_prediction.de")

    # save greedy search decoder data
    ut.save_list_as_txt(
        path,
        map(lambda x: [dic_keys[i] for i in x], greedy_values),
    )
    # beam_text = beam_decoder(pred_values, 3)
    # create_text_files(beam_text, k=3)
    return path


def inner_beam(test_model, i, s, t, k):
    # nonlocal file_txt, test_model
    t_1, t_2 = 0, 0
    # i, (s, t) in enumerate(zip(source, target)):

    if i == 10:
        return
    batch = Batch()
    batch = create_batch(batch, s, t)
    candidate_sentences = [[[0], 0.0]]
    pred_values = []
    test_model = get_model(test_model)
    for iterator in range(len(batch.source)):
        all_candidates = []
        for j in range(len(candidate_sentences)):
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
            for l in range(len(k_best)):
                candidate = [
                    seq + [get_value(k_best.indices[l])],
                    score - math.log(get_value(k_best.values[l])),
                ]
                all_candidates.append(candidate)
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        candidate_sentences = ordered[:k]
    return candidate_sentences


def beam_decoder(test_model, source, target, k):
    file_txt = []

    # open pool for multiprocessing library
    with multiprocessing.Pool(processes=8) as p:
        # multiprocessing lines
        file_txt = p.starmap(
            inner_beam,
            zip(
                itertools.repeat(test_model),
                range(10),
                source[:10],
                target[:10],
                itertools.repeat(k),
            ),
        )
    p.close()
    p.join()
    for e in file_txt:
        print(e)
    save_k_txt(file_txt, k)


def save_k_txt(file_txt, k):
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


def calc_scores(test_model, source, target):
    test_model = get_model(test_model)
    scores = []
    t_1, t_2 = 0, 0

    #
    for i, (s, t) in enumerate(zip(source, target)):
        if i == 1:
            break
        score = 0
        batch = Batch()
        batch = create_batch(batch, s, t)
        print(i)

        pred_values = []
        for iterator in range(len(batch.source)):
            act_tar_label = batch.label[iterator]
            dic = {
                "I0": np.array([batch.source[iterator]]),
                "I1": np.array([[t_1, t_2]]),
            }
            # prediction step
            pred_values.append(test_model.predict(dic, batch_size=1, callbacks=None)[0])
            # print(pred_values)
            # swap values and get the best predictions
            t_1 = t_2
            t_2 = act_tar_label
            # print(pred_values[iterator])
            score += math.log(get_value(pred_values[iterator][act_tar_label]))
        scores.append(score)

    dic_keys = dic_tar.get_keys()
    tmp = list(map(lambda x: [dic_keys[i] for i in x], target[:10]))

    scores = list(map(lambda x: math.exp(x), scores))

    for i in range(len(tmp)):
        tmp[i].append(str(scores[i]))

    # save greedy search decoder data
    ut.save_list_as_txt(
        os.path.join(cur_dir, "a1_scores", "scores.de"),
        tmp,
    )


#
def loader(model, val_src, val_tar, window=2, mode="b"):
    """
    Load and test the model
    """
    src = ut.read_from_file(val_src)
    tar = ut.read_from_file(val_tar)

    source, target = get_word_index(src, tar)
    # batch = get_all_batches(source, target, window)

    if mode == "b":
        return beam_decoder(model, source, target, 3)  # use beam search
    if mode == "g":
        return greedy_decoder(model, source, target)  # use greedy search
    if mode == "s":
        return calc_scores(model, source, target)  # calculate Score


def main():
    # read learned dictionaries for source and target
    dic_src.get_stored(os.path.join(cur_dir, "dictionaries", "source_dictionary"))
    dic_tar.get_stored(os.path.join(cur_dir, "dictionaries", "target_dictionary"))

    # z = [
    #     os.path.join(os.curdir, "predictions", "beam_prediction_k=1_.en")
    #     for i in range(1)
    # ]

    # compare_bleu_scores(os.path.join(cur_dir, "test_data", "multi30k.dev.en"), z)

    # load model and predict outputs
    loader(
        "training_1/train_model.epoch11-loss0.50.hdf5",
        os.path.join(cur_dir, "output", "multi30k.dev_subword.en"),
        os.path.join(cur_dir, "output", "multi30k.dev_subword.de"),
        mode="b",
    )


if __name__ == "__main__":
    # data = [
    #     [0.1, 0.2, 0.3, 0.4, 0.5],
    #     [0.5, 0.4, 0.3, 0.2, 0.1],
    #     [0.1, 0.2, 0.3, 0.4, 0.5],
    #     [0.5, 0.4, 0.3, 0.2, 0.1],
    #     [0.1, 0.2, 0.3, 0.4, 0.5],
    #     [0.5, 0.4, 0.3, 0.2, 0.1],
    #     [0.1, 0.2, 0.3, 0.4, 0.5],
    #     [0.5, 0.4, 0.3, 0.2, 0.1],
    #     [0.1, 0.2, 0.3, 0.4, 0.5],
    #     [0.5, 0.4, 0.3, 0.2, 0.1],
    #     [0.5, 0.4, 0.3, 0.2, 0.1],
    #     [0.5, 0.4, 0.3, 0.2, 0.1],
    #     [0.5, 0.4, 0.3, 0.2, 0.1],
    # ]
    # print(data)

    # out = beam_decoder(np.array(data), 3)
    # print(out)
    main()
