from tensorflow.python.training.saving import checkpoint_options
from encoder import run_bpe
import tensorflow as tf
import numpy as np
from tensorflow._api.v2 import data
from tensorflow.python.ops.gen_dataset_ops import dataset_to_graph_v2
import utility as ut
import os
from tensorflow.python.keras.layers.core import Dense
import batches
from batches import Batch, get_next_batch, get_all_batches
from tensorflow.keras.layers import Input, Concatenate, Embedding
from tensorflow.keras.models import Model
import sys

# globals sit here.
from custom_model import FeedForward, ExtCallback
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


def train_by_fit(train_model, dataset):
    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_path = os.path.join(cur_dir, checkpoint_path)

    # sys.arg[1] tells python to print training reports every 10 batches
    call_for_metrics = ExtCallback(int(sys.argv[1]))
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, save_weights_only=False, verbose=1
    )
    history = train_model.model.fit(
        dataset,
        epochs=5,
        callbacks=[cp_callback],
        batch_size=200,
        # verbose=0,
        # validation_data=(input_list, output_tar),
    )
    return history


def run_nn(sor_file, tar_file, window=2):
    batch = Batch()
    # BUG: REQ das Label muss während das lernen immer bekannt sein. S9 Architektur in letzte VL

    # store source and target file as list of words
    src = ut.read_from_file(sor_file)
    trg = ut.read_from_file(tar_file)

    # get word mapping for both source and index files
    source, target = batches.get_word_index(src, trg)
    batch = get_all_batches(source, target, window)

    # Modell is a sub class from keras.Model()
    # Modell() in custom_model.py
    train_model = FeedForward()
    train_model.build_model(window)
    train_model.show_summary()
    train_model.compile_model()

    #     # creates tensors from lists
    feed_src = np.array(batch.source)
    feed_tar = np.array(batch.target)

    #     feed_src = feed_src.map(lambda x: tf.one_hot(x, depth=len(dic_src)))
    #     feed_tar = feed_tar.map(lambda x: tf.one_hot(x, depth=len(dic_tar)))
    #     feed_zip = tf.data.Dataset.zip((feed_src,feed_tar))
    output_tar = []
    # for step, elem in enumerate(batch.label):
    #     output_tar.append(tf.one_hot(elem, depth=len(dic_tar)))
    output_tar = np.array(batch.label)

    #     output_tar = tf.reshape(output_tar, (200, 1, len(dic_tar) + len(dic_src)))
    #     input_src = np.array(feed_src)
    #     input_tar = np.array(feed_tar)

    #     # dictionary to specify inputs at each input point in NN
    input_list = {"I0": feed_src, "I1": feed_tar}
    dataset = tf.data.Dataset.from_tensor_slices(input_list)

    # loading batches to dataset
    data_set = tf.data.Dataset.from_tensor_slices(output_tar)
    dataset = tf.data.Dataset.zip((dataset, data_set)).batch(200, drop_remainder=True)

    # run nn training with fit
    history = train_by_fit(train_model, dataset)

    # print the returned metrics from our method
    print(history.history)


def integrate_gpu():
    """
    Method to check whether gpu should remain integrated
    """
    if not _LOCAL_DEVICES:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def main():
    # TODO NEXT: function for hyperparameters?

    # check local devices
    integrate_gpu()

    # running BPE with 7k operations on dev text
    # DONE: this part has been previously done! uncomment
    #       next line to create files for subsword split.
    # run_bpe(7000)

    ## Run neural network
    run_nn(
        os.path.join(cur_dir, "output", "multi30k_subword.en"),
        os.path.join(cur_dir, "output", "multi30k_subword.de"),
    )


main()