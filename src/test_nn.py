"""
    ######################## Adapted Architecture ########################
    # input layer: src window - target window
    # feste Größe N=200
    # zur Darstellung des Vokabulars wird one hot vector benötigt
    # Fully connected source  - fully connected target
    # concat layer
    # fully connected layer 1
    # fully connected layer 2 / Projektion
    # Ausgabelayer: softmax layer.
"""
import math
import os
import sys
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.backend import _LOCAL_DEVICES
from kerastuner.tuners import RandomSearch
import batches
from batches import Batch, get_all_batches
import utility as ut

# from encoder import run_bpe
# from dictionary import dic_src, dic_tar
import config_custom_train as config

# remove us
import glob
import decoder

# globals sit here.
from custom_model import (
<<<<<<< HEAD:src/test_nn.py
    BleuCallback,
    MetricsCallback,
    WordLabelerModel,
    build_search_model,
=======
    MetricsCallback,
    WordLabelerModel,
    build_search_model,
    ModelCheckpoint,
>>>>>>> 60332a317c1af37fbd460f82d930e957d4cc1d67:Abgabe 3/test_nn.py
)
from utility import cur_dir

# NOTE: using loop to do the feed forward is slow because loops in py are inefficient.
#   -> use %timeit !?


def get_callback_list(cp_freq=1, tb_vis=False, lr_frac=False, is_val=False):
    """
    Returns list of callback for validation and training
    """
    # specify path to save checkpoint data and tensorboard
    checkpoint_path = "training_1/train_model.epoch{epoch:02d}-loss{loss:.2f}.hdf5"
    tb_log = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_log = os.path.join(cur_dir, tb_log)
    checkpoint_path = os.path.join(cur_dir, checkpoint_path)

    bleu_callback = BleuCallback(cp_freq)

    # sys.arg[1] tells python to print training reports every n batches
    # specify callbacks to store metrics and logs
    call_for_metrics = MetricsCallback(int(sys.argv[1]))

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        monitor="perplexity",
        filepath=checkpoint_path,
        save_weights_only=False,
        verbose=1,
        save_freq=int(cp_freq),
    )

    # callback for reducing the learningrate if metrics stagnate on validation data
    learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(
        # monitor defines which metric we are monitoring
        monitor="val_accuracy",
        # how many evaluations of no improvement do we wait until we change the LR (learning rate)
        patience=1,
        verbose=1,
        # factor to cut the LR in half
        factor=0.5,
        # how often should the LR be reduced? -> give minimal LR here:
        min_lr=0.0001,
    )
    # callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        # monitor defines which metric we are monitoring
        monitor="val_accuracy",
        # how many evaluations of no improvement do we wait until we change the LR (learning rate)
        patience=4,
        restore_best_weights=True,
    )

    # tensorboard callback
    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=tb_log,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq=50,
        embeddings_freq=1,
    )

<<<<<<< HEAD:src/test_nn.py
    callback_list = [
        call_for_metrics,
        early_stopping,
        cp_callback,
        bleu_callback,
    ]
=======
    callback_list = [call_for_metrics, early_stopping, cp_callback]
>>>>>>> 60332a317c1af37fbd460f82d930e957d4cc1d67:Abgabe 3/test_nn.py

    if lr_frac:
        callback_list.append(learning_rate_reduction)

    if tb_vis:
        if not is_val:
            try:
                os.system("rm -rf ./logs/")
            except:
                print("No previous logs...")
        callback_list.append(tb_callback)

    # creating callback list for fit()
    return callback_list


def validate_by_evaluate(train_model, val_data):
    """
    Runs evaluate on train_model
    """
    # creating callback list for evaluate()
    callback_list = get_callback_list(
        cp_freq=sys.argv[7], tb_vis=sys.argv[9], is_val=True
    )

    # run evaluate()
    history = train_model.evaluate(
        val_data,
        callbacks=callback_list,
        batch_size=200,
        verbose=0,
    )
    return history


def train_by_fit(train_model, dataset_train, dataset_val):
    """
    Runs fit() on train_model
    """
    # creating callback list for fit()
    callback_list = get_callback_list(
        cp_freq=sys.argv[7], lr_frac=sys.argv[8], tb_vis=sys.argv[9], is_val=False
    )

    # run fit()
    history = train_model.fit(
        dataset_train,
        epochs=18,
        callbacks=callback_list,
        verbose=0,
        validation_data=dataset_val,
        use_multiprocessing=True,
        workers=7,
    )
    return history


def run_nn(sor_file, tar_file, val_src, val_tar, window=2, val_on_dev=False):
    """
    Trains and validates training data.
    """
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
    dataset = tf.data.Dataset.zip((dataset, data_set)).batch(2000, drop_remainder=True)

    # preprocessing data
    batch_count = math.floor(batch.size / 2000)
    batch_count_train = int(batch_count * 0.9)
    dataset.shuffle(int(batch_count * 1.1))

    if not val_on_dev:
        dataset_train = dataset.take(batch_count_train)  # training data
        dataset_val = dataset.skip(batch_count_train)  # validation data

    #### begin validation
    val_batch = get_all_batches(val_source, val_target, window)
    val_feed_src = np.array(val_batch.source)
    val_feed_tar = np.array(val_batch.target)
    val_output_tar = np.array(val_batch.label)

    ## dictionary to specify inputs at each input point in NN
    val_input_list = {"I0": val_feed_src, "I1": val_feed_tar}
    val_dataset = tf.data.Dataset.from_tensor_slices(val_input_list)

    # loading batches to dataset
    val_data_set = tf.data.Dataset.from_tensor_slices(val_output_tar)
    val_dataset = tf.data.Dataset.zip((val_dataset, val_data_set)).batch(
        2000, drop_remainder=True
    )

    # run nn training with fit
    history = train_by_fit(
        train_model,
        dataset if val_on_dev else dataset_train,
        val_dataset if val_on_dev else dataset_val,
    )

    # save model
    # train_model.save("model.h5")
    # print("**** Saved model to disk (after training) ****")

    # print the returned metrics from our method
    # end of training
    print(history.history)

    # evaluate results
    val_history = validate_by_evaluate(train_model, val_dataset)
    print(val_history)

    save_name = f"model_val-loss-{val_history[0]:.3f}_val-acc-{val_history[1]:.3f}_val-per-{val_history[2]:.3f}"
    train_model.save(save_name + ".h5")
    print("**** Saved model to disk (after eval) ****")


def start_hp_search(tuner, dataset_train, dataset_val):
    """
    Runs fit() on train_model
    """
    # creating callback list for search()
    callback_list = get_callback_list(
        cp_freq=sys.argv[7], lr_frac=sys.argv[8], tb_vis=sys.argv[9], is_val=False
    )

    # run search()
    # start search
    history = tuner.search(
        dataset_train,
        epochs=config.search_params["epochs"],
        validation_data=dataset_val,
        callbacks=callback_list,
        verbose=0,
    )

    return history, tuner


def hyperparam_search(sor_file, tar_file, val_src, val_tar, window=2, val_on_dev=True):
    """
    Starts hyperparameter search, saves best model, best params and evaluates best model
    """
    tf.config.list_physical_devices("GPU")
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

    # initialize tuner
    tuner = RandomSearch(
        build_search_model,
        objective="val_accuracy",
        max_trials=config.search_params["max_trials"],
        executions_per_trial=config.search_params["executions_per_trial"],
        directory=config.search_params["directory_name"],
        project_name="hps_feedforward_translator",
    )

    # print search space summary
    print(tuner.search_space_summary())

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
    dataset = tf.data.Dataset.zip((dataset, data_set)).batch(2000, drop_remainder=True)

    #### validation data preprocessing
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
        2000, drop_remainder=True
    )

    # preprocessing data
    batch_count = math.floor(batch.size / 2000)
    batch_count_train = int(batch_count * 0.9)
    dataset.shuffle(int(batch_count * 1.1))

    if not val_on_dev:
        dataset_train = dataset.take(batch_count_train)  # training data
        dataset_val = dataset.skip(batch_count_train)  # validation data

    # start hp search
    history, tuner = start_hp_search(
        tuner,
        dataset if val_on_dev else dataset_train,
        val_dataset if val_on_dev else dataset_val,
    )

    # get best model(s) and hyperparameters from search
    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.save("best_model/best_model.h5")
    best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
    print("\n------------ Best params: ------------")
    print(best_hyperparameters)

    # TODO: Save best_hps to file that updates every time someone gets better params

    # print the returned metrics from our method
    # end of search
    print("\n------------ Results summary: ------------")
    print(tuner.results_summary())
    print("\n------------ History: ------------")
    print(history.history)

    # evaluate results
    val_history = validate_by_evaluate(best_model, val_dataset)
    print(val_history)


def integrate_gpu():
    """
    Method to check whether gpu should remain integrated
    """


def main():
    """
    main function
    """
    # TODO NEXT: function for hyperparameters?

    # check local devices
    print(
        "-" * 50,
        "\nNum GPUs Available: ",
        len(tf.config.experimental.list_physical_devices("GPU")),
        "\n" + "-" * 50,
    )
<<<<<<< HEAD:src/test_nn.py
    if not (sys.argv[11].lower() == "true"):
=======
    if sys.argv[11].lower() == "true":
>>>>>>> 60332a317c1af37fbd460f82d930e957d4cc1d67:Abgabe 3/test_nn.py
        integrate_gpu()

    # running BPE with 7k operations on dev text
    # DONE: this part has been previously done! uncomment
    #       next line to create files for subsword split.
    # run_bpe(sys.args[2]) # system argument holds number of operations in bpe

    ## Run neural network
    if sys.argv[10].lower() == "true":
        print("----------------- Starting hyperparametersearch ... -----------------")
        hyperparam_search(
            os.path.join(cur_dir, "output", sys.argv[3]),
            os.path.join(cur_dir, "output", sys.argv[4]),
            os.path.join(cur_dir, "output", sys.argv[5]),
            os.path.join(cur_dir, "output", sys.argv[6]),
            val_on_dev=(sys.argv[11].lower() == "true"),
        )
    else:
        run_nn(
            os.path.join(cur_dir, "output", sys.argv[3]),
            os.path.join(cur_dir, "output", sys.argv[4]),
            os.path.join(cur_dir, "output", sys.argv[5]),
            os.path.join(cur_dir, "output", sys.argv[6]),
            val_on_dev=(sys.argv[11].lower() == "true"),
        )


def outdated_tester():
    """used to test"""
    path = os.path.join(cur_dir, "training_1")
    dev_1 = os.path.join(cur_dir, "output", sys.argv[5])
    dev_2 = os.path.join(cur_dir, "output", sys.argv[6])
    max_file = max(glob.glob(path + "/*hdf5"), key=os.path.getctime)
    data_path = decoder.loader(max_file, dev_1, dev_2, mode="g")
    print(data_path)


if __name__ == "__main__":
    main()
    # l()
