from tensorflow.python.training.saving import checkpoint_options
from encoder import run_bpe
import tensorflow as tf
import numpy as np
from tensorflow._api.v2 import data
from tensorflow.python.ops.gen_dataset_ops import dataset_to_graph_v2
import utility as ut
import os
import batches
from batches import Batch, get_all_batches
import sys
import math
import datetime

# globals sit here.
from custom_model import MetricsCallback, Perplexity, WordLabelerModel
from dictionary import dic_src, dic_tar
from utility import cur_dir
from tensorflow.python.keras.backend import _LOCAL_DEVICES

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


def validate_by_evaluate(train_model, val_data, cp_freq=1000, tb_vis=False):
    # specify path to save checkpoint data and tensorboard
    checkpoint_path = "training_1/cp_val.hdf5"
    tb_log = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_log = os.path.join(cur_dir, tb_log)
    checkpoint_path = os.path.join(cur_dir, checkpoint_path)

    # sys.arg[1] tells python to print training reports every n batches
    # specify callbacks to store metrics and logs
    call_for_metrics = MetricsCallback((sys.argv[1]))

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1,
        save_freq=cp_freq,
    )

    # callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        # monitor defines which metric we are monitoring
        monitor="accuracy",
        # how many evaluations of no improvement do we wait until we change the LR (learning rate)
        patience=3,
        restore_best_weights=True,
    )

    # tensorboard callback
    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=tb_log,
        histogram_freq=1,
    )

    # creating callback list for fit()
    callback_list = [call_for_metrics, early_stopping, cp_callback]

    if tb_vis:
        try:
            os.system("rm -rf ./logs/")
        except:
            print("No previous logs...")
        callback_list.append(tb_callback)

    # run fit()
    history = train_model.evaluate(
        val_data,
        callbacks=callback_list,
        batch_size=200,
        verbose=0,
    )
    return history


def train_by_fit(
    train_model, dataset_train, dataset_val, cp_freq=1000, lr_frac=False, tb_vis=False
):
    # specify path to save checkpoint data and tensorboard
    checkpoint_path = "training_1/train_model.epoch{epoch:02d}-loss{loss:.2f}.hdf5"
    tb_log = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_log = os.path.join(cur_dir, tb_log)
    checkpoint_path = os.path.join(cur_dir, checkpoint_path)

    # sys.arg[1] tells python to print training reports every 10 batches
    # specify callbacks to store metrics and logs
    call_for_metrics = MetricsCallback(int(sys.argv[1]))

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        monitor="perplexity",
        filepath=checkpoint_path,
        save_weights_only=False,
        verbose=1,
        save_freq=cp_freq,
    )

    # callback for reducing the learningrate if metrics stagnate on validation data
    learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(
        # monitor defines which metric we are monitoring
        monitor="accuracy",
        # how many evaluations of no improvement do we wait until we change the LR (learning rate)
        patience=1,
        verbose=1,
        # factor to cut the LR in half
        factor=0.5,
        # how often should the LR be reduced? -> give minimal LR here:
        min_lr=0.00001,
    )
    # callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        # monitor defines which metric we are monitoring
        monitor="accuracy",
        # how many evaluations of no improvement do we wait until we change the LR (learning rate)
        patience=3,
        restore_best_weights=True,
    )

    # tensorboard callback
    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=tb_log,
        histogram_freq=1,
    )

    # creating callback list for fit()
    callback_list = [call_for_metrics, early_stopping, cp_callback]

    if lr_frac:
        callback_list.append(learning_rate_reduction)

    if not tb_vis:
        try:
            os.system("rm -rf ./logs/")
        except:
            print("No previous logs...")
        callback_list.append(tb_callback)

    # run fit()
    history = train_model.fit(
        dataset_train,
        epochs=11,
        callbacks=callback_list,
        batch_size=200,
        verbose=0,
        validation_data=dataset_val,
    )
    return history


def run_nn(sor_file, tar_file, val_src, val_tar, window=2):
    batch = Batch()
    # BUG: REQ das Label muss während das lernen immer bekannt sein. S9 Architektur in letzte VL

    # store source and target file as list of words
    src = ut.read_from_file(sor_file)
    trg = ut.read_from_file(tar_file)
    # store source and target file as list of words
    val_src = ut.read_from_file(val_src)
    val_trg = ut.read_from_file(val_tar)

    # get word mapping for both training files and index files
    source, target = batches.get_word_index(src, trg)
    batch = get_all_batches(source, target, window)

    # get word mapping for both validation files and index files
    val_source, val_target = batches.get_word_index(val_src, val_trg)

    # Modell is a sub class from keras.Model()
    # Modell() in custom_model.py
    train_model = WordLabelerModel()
    # train_model.build_model(window)

    train_model.compile_model()
    print(train_model.summary())

    #  creates tensors from lists
    feed_src = np.array(batch.source)
    feed_tar = np.array(batch.target)

    output_tar = []
    output_tar = np.array(batch.label)

    # train_model.model = tf.keras.models.load_model(
    #     "training_1/cp.ckpt", custom_objects={"perplexity": Perplexity}
    # )

    ## dictionary to specify inputs at each input point in NN
    input_list = {"I0": feed_src, "I1": feed_tar}
    dataset = tf.data.Dataset.from_tensor_slices(input_list)

    # loading batches to dataset
    data_set = tf.data.Dataset.from_tensor_slices(output_tar)
    dataset = tf.data.Dataset.zip((dataset, data_set)).batch(200, drop_remainder=True)

    # preprocessing data
    batch_count = math.floor(batch.size / 200)
    batch_count_train = int(batch_count * 0.9)
    dataset.shuffle(int(batch_count * 1.1))
    dataset_train = dataset.take(batch_count_train)  # training data
    dataset_val = dataset.skip(batch_count_train)  # validation data

    # run nn training with fit
    history = train_by_fit(train_model, dataset_train, dataset_val, tb_vis=True)

    # print the returned metrics from our method
    # end of training
    print(history.history)

    #### begin validation
    val_batch = get_all_batches(val_source, val_target, window)
    val_feed_src = np.array(val_batch.source)
    val_feed_tar = np.array(val_batch.target)
    val_output_tar = []
    val_output_tar = np.array(val_batch.label)

    ## dictionary to specify inputs at each input point in NN
    val_input_list = {"I0": val_feed_src, "I1": val_feed_tar}
    val_dataset = tf.data.Dataset.from_tensor_slices(val_input_list)

    # loading batches to dataset
    val_data_set = tf.data.Dataset.from_tensor_slices(val_output_tar)
    val_dataset = tf.data.Dataset.zip((val_dataset, val_data_set)).batch(
        200, drop_remainder=True
    )

    # evaluate results
    val_history = validate_by_evaluate(train_model, val_dataset)
    print(val_history)


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
        os.path.join(cur_dir, "output", sys.argv[3]),
        os.path.join(cur_dir, "output", sys.argv[4]),
        os.path.join(cur_dir, "output", sys.argv[5]),
        os.path.join(cur_dir, "output", sys.argv[6]),
    )


main()