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
from tensorflow.python.framework.tensor_conversion_registry import get
from tensorflow.python.ops.gen_math_ops import mod
from tensorflow_addons.seq2seq import decoder
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
import tensorflow_addons as tfa


def get_model(source, batch_size=1):
    """builds and returns a model from model's path"""
    enc, dec = get_enc_dec_paths()
    test_model = rnn.Translator(len(dic_tar), len(dic_src), 200, 200, batch_size)

    # initialize layers
    enc_output, h, c = test_model.encoder(
        source, test_model.encoder.initialize_hidden_state()
    )

    test_model.decoder.attention_mechanism.setup_memory(enc_output)

    dec_init = test_model.decoder.build_initial_state(batch_size, [h, c], tf.float32)

    test_model.decoder(tf.zeros((batch_size, 47)), dec_init)

    test_model.encoder.load_weights(enc)
    test_model.decoder.load_weights(dec)

    # compile then return loaded model
    test_model.compile(
        optimizer="adam",
    )
    return test_model, enc_output, h, c


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


def translate_line(i, source, k):
    """used by beam decoder to process lines simultaneously"""
    # nonlocal file_txt, test_model
    model, enc_output, h, c = get_model(source)

    enc_output = tfa.seq2seq.tile_batch(enc_output, k)

    model.decoder.attention_mechanism.setup_memory(enc_output)

    hidden_state = tfa.seq2seq.tile_batch([h, c], k)

    decoder_init_state = model.decoder.rnn_cell.get_initial_state(
        batch_size=k, dtype=tf.float32
    )

    decoder_init_state = decoder_init_state.clone(cell_state=hidden_state)

    # beam decoder
    beam_instance = tfa.seq2seq.BeamSearchDecoder(
        model.decoder.rnn_cell,
        beam_width=k,
        output_layer=model.decoder.fc,
        maximum_iterations=47,
    )
    inference_decoder_output = tfa.seq2seq.dynamic_decode(
        beam_instance, impute_finished=False
    )[0]

    # embedding_matrix = model.decoder.embedding.variables[0]

    # output, _, _ = beam_instance(
    #     embedding_matrix,
    #     start_tokens=[0],
    #     end_token=14,
    #     initial_state=decoder_init_state,
    # )
    # print("iam here")
    # final_outputs = tf.transpose(output.predicted_ids, perm=(0, 2, 1))
    # beam_scores = tf.transpose(output.beam_search_decoder_output.scores, perm=(0, 2, 1))

    return inference_decoder_output  # final_outputs.numpy(), beam_scores.numpy()


def beam_decoder(source, k):
    """finds the best translation scores using the beam decoder."""
    file_txt = []

    # open pool for multiprocessing library
    with multiprocessing.Pool(processes=8) as pool:
        # multiprocessing lines
        file_txt = pool.starmap(
            translate_line,
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


def rnn_pred_batch(source_list, target_list):
    """returns a batch from source file by padding all sentences"""
    source_list, _ = get_word_index(source_list, target_list)
    source_list = list(map(lambda x: [0] + list(x) + [1], source_list))

    # instead of maxlen=46 use max_word_in_line
    return tf.keras.preprocessing.sequence.pad_sequences(
        source_list, maxlen=47, value=1, padding="post"
    )


def get_enc_dec_paths():
    """returns encoder and decoder path as tuple"""
    enc_path = os.path.join(cur_dir, "rnn_checkpoints", "encoder.epoch01-loss1.51.hdf5")
    dec_path = os.path.join(cur_dir, "rnn_checkpoints", "decoder.epoch01-loss1.51.hdf5")

    return (enc_path, dec_path)


def evaluate_sentence(sentence):
    """evals sentence"""
    print(sentence)
    model = get_model(sentence)[0]

    inputs = tf.convert_to_tensor(sentence)
    inference_batch_size = inputs.shape[0]
    print(inference_batch_size)
    result = ""

    enc_start_state = [
        tf.zeros((inference_batch_size, 200)),
        tf.zeros((inference_batch_size, 200)),
    ]
    enc_out, enc_h, enc_c = model.encoder(inputs, enc_start_state)
    start_tokens = tf.fill([inference_batch_size], 1)
    end_token = 2

    greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler()

    # Instantiate BasicDecoder object
    decoder_initial_state = model.decoder.build_initial_state(
        inference_batch_size, [enc_h, enc_c], tf.float32
    )
    decoder_embedding_matrix = model.decoder.embedding.variables[0]

    decoder_instance = tfa.seq2seq.BasicDecoder(
        cell=model.decoder.rnn_cell,
        sampler=greedy_sampler,
        output_layer=model.decoder.fc,
        maximum_iterations=47,
    )
    # Setup Memory in decoder stack
    model.decoder.attention_mechanism.setup_memory(enc_out)

    outputs, _, _ = decoder_instance(
        decoder_embedding_matrix,
        start_tokens=start_tokens,
        end_token=end_token,
        initial_state=decoder_initial_state,
    )
    return outputs.sample_id.numpy()


def main():
    """main method"""
    rnn.init_dics()
    source = read_from_file(
        os.path.join(cur_dir, "test_data", "multi30k.dev_subword.de")
    )
    target = read_from_file(
        os.path.join(cur_dir, "test_data", "multi30k.dev_subword.en")
    )

    inputs = rnn_pred_batch(
        ["eine gruppe von männern lädt baum@@ wolle auf einen lastwagen ."], target
    )
    print(evaluate_sentence(inputs))
    # inputs = tf.convert_to_tensor(inputs)
    # print(inputs)
    # f, s = translate_line(1, inputs, 1)
    # print(f, s)


def rec_dec_tester():
    """called for testing specific methods"""
    rnn.init_dics()
    m = get_model(None)


if __name__ == "__main__":
    main()
    # rec_dec_tester()
