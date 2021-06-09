"""

"""
from sys import exit
from tensorflow.keras import Model
from tensorflow.python.keras.backend import argmax
from batches import *
from custom_model import Perplexity, WordLabelerModel
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
    result = [[] for _ in range(k)]
    sentence = [[] for _ in range(k)]
    keys_list = dic_tar.get_keys()

    for elem in arr:
        k_candidates = tf.math.top_k(elem, k).indices
        k_candidates = [tf.keras.backend.get_value(e) for e in k_candidates]
        # print("k_candidates:", k_candidates)
        for i in range(k):
            sentence[i].append(keys_list[int(k_candidates[i])])
            if keys_list[int(k_candidates[i])] == "." and sentence[i] != []:
                result[i].append(sentence[i])
                # print("result:", result)
                sentence[i] = []

    #
    return result


#
def tester(sor_file, tar_file, val_src, val_tar, window=2):
    """
    Load and test the model
    """
    batch = Batch()

    # store source and target file as list of words
    src = ut.read_from_file(sor_file)
    trg = ut.read_from_file(tar_file)
    val_src = ut.read_from_file(val_src)
    val_tar = ut.read_from_file(val_tar)
    # get word mapping for both training files and index files
    get_word_index(src, trg)
    # Needed for dictionary size TODO Maybe?
    source, target = get_word_index(val_src, val_tar)
    batch = get_all_batches(source, target, window)

    # test_model = WordLabelerModel()
    test_model = tf.keras.models.load_model(
        "training_101/train_model.epoch01-loss3.94.hdf5",
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

    greedy_text = greedy_decoder(history)
    beam_text = beam_decoder(history, 3)

    ut.save_list_as_txt(
        os.path.join(cur_dir, "predictions", "greedy_prediction.de"),
        greedy_text,
    )
    for i in range(3):
        ut.save_list_as_txt(
            os.path.join(cur_dir, "predictions", "greedy_prediction" + str(i) + ".de"),
            beam_text[i],
        )


def main():
    tester(
        os.path.join(cur_dir, "output", "multi30k_subword.en"),
        os.path.join(cur_dir, "output", "multi30k_subword.de"),
        os.path.join(cur_dir, "output", "multi30k.dev_subword.en"),
        os.path.join(cur_dir, "output", "multi30k.dev_subword.de"),
    )


if __name__ == "__main__":
    main()
