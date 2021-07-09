"""
recurrent_dec.py offers methods to compute beam search and greedy search 
translations are made using the RNN model from recurrent_nn.py
"""
import os
import tensorflow as tf
import numpy as np
from encoder import revert_bpe
import utility as ut
import recurrent_nn as rnn
from dictionary import dic_tar, dic_src
from utility import cur_dir, read_from_file
import tensorflow_addons as tfa


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


def roll_out_encoder(sentence, batch_size=1):
    """builds and returns a model from model's path"""
    enc, dec = get_enc_dec_paths()
    test_model = rnn.Translator(len(dic_tar), len(dic_src), 200, 200, batch_size)
    dec_input = [dic_src.bi_dict["<s>"], 0 for _ in range(len(sentence)-1)]
    dec_input = np.array(dec_input)

    enc_output, _, _ = test_model.encoder(sentence, None)
    dec_output = test_model.decoder((dec_input, enc_output), None) 

    test_model.load_weights(enc)
    test_model.load_weights(dec)

    return test_model, enc_output, dec_output


def translate_sentence(sentence, k):
    """translates sentence using beam search algorithm"""
    model, enc_output, dec_output = roll_out_encoder(sentence)
    e_1 = tf.math.top_k(dec_output)


def rnn_pred_batch(source_list, target_list):
    """returns a batch from source file by padding all sentences"""
    for i, _ in enumerate(source_list):
        tmp = [
            dic_src.get_index(x) if x in dic_src.bi_dict else 3 for x in source_list[i]
        ]
        source_list[i] = tmp
    source_list = list(map(lambda x: [1] + list(x) + [2], source_list))

    # instead of maxlen=46 use max_word_in_line
    return tf.keras.preprocessing.sequence.pad_sequences(
        source_list, maxlen=47, value=0, padding="post"
    )


def get_enc_dec_paths():
    """returns encoder and decoder path as tuple"""
    enc_path = os.path.join(cur_dir, "rnn_checkpoints", "encoder.epoch01-loss1.68.hdf5")
    dec_path = os.path.join(cur_dir, "rnn_checkpoints", "decoder.epoch01-loss1.68.hdf5")

    return (enc_path, dec_path)


def main():
    """main method"""
    tfa.options.disable_custom_kernel()
    rnn.init_dics()
    source = read_from_file(
        os.path.join(cur_dir, "test_data", "multi30k.dev_subword.de")
    )
    target = read_from_file(
        os.path.join(cur_dir, "test_data", "multi30k.dev_subword.en")
    )

    inputs = rnn_pred_batch(
        ["ein mann schläft in einem grünen raum auf einem sofa ."], target
    )
    keys = dic_tar.get_keys()
    
    
    # inputs = tf.convert_to_tensor(inputs)
    # print(inputs)
    # f, s = translate_line(1, inputs, 1)
    # print(f, s)


def rec_dec_tester():
    """called for testing specific methods"""
    rnn.init_dics()
    m = roll_out_encoder(None)


if __name__ == "__main__":
    main()
    # rec_dec_tester()
