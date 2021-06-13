"""

"""
from metrics import compare_bleu_scores
from encoder import rename_me
from sys import exit
from tensorflow.keras import Model
from tensorflow.python.keras.backend import argmax
from batches import *
from custom_model import Perplexity, WordLabelerModel
from tensorflow.keras.backend import get_value
import tensorflow as tf
import numpy as np
import utility as ut


def greedy_decoder_outdated(arr):
    """ Implements the greedy decoder search algorithm """
    tmp = []
    keys_list = dic_tar.get_keys()

    result = []
    for elem in arr:
        biggest = tf.keras.backend.get_value(argmax(elem))
        tmp.append(keys_list[biggest])
        if biggest == dic_tar.bi_dict["."] and tmp != []:
            result.append(tmp)
            tmp = []

    return result


def beam_decoder_outdated(data, k):
    """
    Implements the beam search decoding algorithm
    Returns k lists with best possible predictions
    """

    # avrg line length
    # wc -lw data_exercise_3/multi30k.de | awk '{print $2/$1}'

    sequences = [[list(), 0.0]]
    for elem in data:
        all_candidates = list()
        k_candidates = tf.math.top_k(elem, k=k)
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(k_candidates)):
                candidate = [
                    seq + [get_value(k_candidates.indices[j])],
                    score - math.log(get_value(k_candidates.values[j])),
                ]
                all_candidates.append(candidate)
        # order all candidates by score

        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        # select k best
        sequences = ordered[:k]

    # for elem in arr:
    #     k_candidates = tf.math.top_k(elem, k).indices
    #     k_candidates = [tf.keras.backend.get_value(e) for e in k_candidates]

    #     # print("k_candidates:", k_candidates)
    #     for i in range(k):
    #         sentence[i].append(keys_list[int(k_candidates[i])])
    #         if keys_list[int(k_candidates[i])] == "." and sentence[i] != []:
    #             result[i].append(sentence[i])
    #             # print("result:", result)
    #             sentence[i] = []

    #
    return sequences


def create_text_files(line, k):
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
    history = []
    greedy_values = []
    t_1, t_2 = 0, 0

    #
    for i, (s, t) in enumerate(zip(source, target)):
        batch = Batch()
        batch = create_batch(batch, s, t)
        print(i)

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

    # save greedy search decoder data
    ut.save_list_as_txt(
        os.path.join(cur_dir, "predictions", "greedy_prediction.de"),
        map(lambda x: [dic_keys[i] for i in x], greedy_values),
    )
    # beam_text = beam_decoder(pred_values, 3)
    # create_text_files(beam_text, k=3)


def beam_decoder(test_model, source, target, k):
    t_1, t_2 = 0, 0

    #
    for i, (s, t) in enumerate(zip(source, target)):
        if i == 20:
            break
        batch = Batch()
        batch = create_batch(batch, s, t)
        candidate_sentences = [[[0], 0.0]]

        pred_values = []
        for iterator in range(len(batch.source)):
            all_candidates = []
            for j in range(len(candidate_sentences)):
                # get last element to compute next prediction
                t_2 = candidate_sentences[j][0][-1]
                print(t_2)
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

        create_text_files(candidate_sentences, k)


def calc_scores(test_model, source, target):
    scores = []
    t_1, t_2 = 0, 0

    #
    for i, (s, t) in enumerate(zip(source, target)):
        if i == 10:
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
def loader(sor_file, tar_file, val_src, val_tar, window=2, mode="b"):
    """
    Load and test the model
    """
    src = ut.read_from_file(val_src)
    tar = ut.read_from_file(val_tar)

    source, target = get_word_index(src, tar)
    # batch = get_all_batches(source, target, window)

    # test_model = WordLabelerModel()
    test_model = tf.keras.models.load_model(
        "training_1/train_model.epoch01-loss3.93.hdf5",
        custom_objects={"WordLabelerModel": WordLabelerModel, "perplexity": Perplexity},
        compile=False,
    )
    print(test_model.summary())
    test_model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        # using categorical cross entropy from keras provided one-hot vectors
        metrics=[
            "accuracy",
            Perplexity(),
        ],
    )

    if mode == "b":
        beam_decoder(test_model, source, target, 3)  # use beam search
    if mode == "g":
        greedy_decoder(test_model, source, target)  # use greedy search
    if mode == "s":
        calc_scores(test_model, source, target)  # calculate Score


def main():
    # read learned dictionaries for source and target
    dic_src.get_stored(os.path.join(cur_dir, "dictionaries", "source_dictionary"))
    dic_tar.get_stored(os.path.join(cur_dir, "dictionaries", "target_dictionary"))

    # z = [
    #     os.path.join(os.curdir, "predictions", "beam_k=1_prediction" + str(i) + ".de")
    #     for i in range(1)
    # ]

    # compare_bleu_scores(os.path.join(cur_dir, "data_exercise_3", "multi30k.dev.de"), z)

    # load model and predict outputs
    loader(
        os.path.join(cur_dir, "output", "multi30k_subword.en"),
        os.path.join(cur_dir, "output", "multi30k_subword.de"),
        os.path.join(cur_dir, "output", "multi30k.dev_subword.en"),
        os.path.join(cur_dir, "output", "multi30k.dev_subword.de"),
        mode="s",
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
