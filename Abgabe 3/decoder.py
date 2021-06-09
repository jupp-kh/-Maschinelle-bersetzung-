"""

"""
from sys import exit
from tensorflow.keras import Model
from tensorflow.python.keras.backend import argmax
from batches import *
from custom_model import Perplexity, WordLabelerModel
from tensorflow.keras.backend import get_value
import tensorflow as tf
import numpy as np
import utility as ut


def greedy_decoder(arr):
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


def beam_decoder(arr, k):
    """
    Implements the beam search decoding algorithm
    Returns k lists with best possible predictions
    """

    line_length = 13  # TODO: no hard
    # avrg line length
    # wc -lw data_exercise_3/multi30k.de | awk '{print $2/$1}'
    lines = []

    while arr.shape[0] >= 13:
        data = arr[:line_length]
        arr = arr[line_length:]
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

        lines.append(sequences)

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
    return lines


def create_text_files(lines, k):
    sentence = [[] for _ in range(k)]
    keys_list = dic_tar.get_keys()

    for line in lines:
        for i in range(k):
            str_lines = map(lambda x: keys_list[x], line[i][0])
            sentence = " ".join(str_lines)
            ut.save_line_as_txt(
                os.path.join(
                    cur_dir, "predictions", "beam_prediction" + str(i) + ".de"
                ),
                sentence,
            )


#
def tester(sor_file, tar_file, val_src, val_tar, window=2):
    """
    Load and test the model
    """
    batch = Batch()
    src = ut.read_from_file(val_src)
    tar = ut.read_from_file(val_tar)

    source, target = get_word_index(src, tar)
    batch = get_all_batches(source, target, window)

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

    # predictions data preprocessing:
    feed_src = np.array(batch.source)
    feed_src = {"I0": feed_src, "I1": np.array(batch.target)}
    feed_src = tf.data.Dataset.from_tensor_slices(feed_src).batch(
        1, drop_remainder=True
    )

    # prediction step
    history = test_model.predict(feed_src, batch_size=1, callbacks=None)

    beam_text = beam_decoder(history, 3)
    create_text_files(beam_text, k=3)

    return

    greedy_text = greedy_decoder(history)

    ut.save_list_as_txt(
        os.path.join(cur_dir, "predictions", "greedy_prediction.de"),
        greedy_text,
    )


def main():
    dic_src.get_stored(os.path.join(cur_dir, "dictionaries", "source_dictionary"))
    dic_tar.get_stored(os.path.join(cur_dir, "dictionaries", "target_dictionary"))
    tester(
        os.path.join(cur_dir, "output", "multi30k_subword.en"),
        os.path.join(cur_dir, "output", "multi30k_subword.de"),
        os.path.join(cur_dir, "output", "multi30k.dev_subword.en"),
        os.path.join(cur_dir, "output", "multi30k.dev_subword.de"),
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
