"""
includes the rnn model
"""
import time
import numpy as np
import os
from keras.layers import Dense
import tensorflow as tf
from tensorflow._api.v2 import train
from dictionary import dic_tar, dic_src
from utility import cur_dir
import batches
import tensorflow_addons as tfa
import recurrent_dec as rnn_dec


# from decoder import loader, greedy_decoder

# import config file for hyperparameter search space
# import config_custom_train as config


class Encoder(tf.keras.Model):
    def __init__(self, dic_size, em_dim, num_units, batch_size, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.num_units = num_units
        self.embedding = tf.keras.layers.Embedding(
            dic_size, em_dim, name="Embedding", mask_zero=True
        )
        self.lstm = tf.keras.layers.LSTM(
            self.num_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
            name="LSTM",
        )

    def call(self, inputs, hidden):
        """implements call from keras.Model"""
        # specify embedding input and pass in embedding in lstm layer
        em = self.embedding(inputs)
        output, h, c = self.lstm(em, initial_state=hidden)
        return output, h, c

    def initialize_hidden_state(self):
        return [
            tf.zeros((self.batch_size, self.num_units)),
            tf.zeros((self.batch_size, self.num_units)),
        ]


class Decoder(tf.keras.Model):
    def __init__(
        self, vocab_size, embedding_dim, dec_units, batch_sz, attention_type="luong"
    ):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.attention_type = attention_type

        # Embedding Layer
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        # Final Dense layer on which softmax will be applied
        self.fc = tf.keras.layers.Dense(vocab_size)

        # Define the fundamental cell for decoder recurrent structure
        self.decoder_rnn_cell = tf.keras.layers.LSTMCell(self.dec_units)

        # Sampler
        self.sampler = tfa.seq2seq.sampler.TrainingSampler()

        # Create attention mechanism with memory = None
        self.attention_mechanism = self.build_attention_mechanism(
            self.dec_units,
            None,
            self.batch_sz * [47 - 1],
            self.attention_type,
        )

        # Wrap attention mechanism with the fundamental rnn cell of decoder
        self.rnn_cell = self.build_rnn_cell(batch_sz)

        # Define the decoder with respect to fundamental rnn cell
        self.decoder = tfa.seq2seq.BasicDecoder(
            self.rnn_cell, sampler=self.sampler, output_layer=self.fc
        )

    def build_rnn_cell(self, batch_sz):
        rnn_cell = tfa.seq2seq.AttentionWrapper(
            self.decoder_rnn_cell,
            self.attention_mechanism,
            attention_layer_size=self.dec_units,
        )
        return rnn_cell

    def build_attention_mechanism(
        self, dec_units, memory, memory_sequence_length, attention_type="luong"
    ):
        # ------------- #
        # typ: Which sort of attention (Bahdanau, Luong)
        # dec_units: final dimension of attention outputs
        # memory: encoder hidden states of shape (batch_size, 47, enc_units)
        # memory_sequence_length: 1d array of shape (batch_size) with every element set to 47 (for masking purpose)

        if attention_type == "bahdanau":
            return tfa.seq2seq.BahdanauAttention(
                units=dec_units,
                memory=memory,
                memory_sequence_length=memory_sequence_length,
            )
        else:
            return tfa.seq2seq.LuongAttention(
                units=dec_units,
                memory=memory,
                memory_sequence_length=memory_sequence_length,
            )

    def build_initial_state(self, batch_sz, encoder_state, Dtype):
        decoder_initial_state = self.rnn_cell.get_initial_state(
            batch_size=batch_sz, dtype=Dtype
        )
        decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
        return decoder_initial_state

    def call(self, inputs, initial_state):
        x = self.embedding(inputs)
        outputs, _, _ = self.decoder(
            x,
            initial_state=initial_state,
            sequence_length=self.batch_sz * [47 - 1],
        )
        return outputs


class Translator(tf.keras.Model):
    def __init__(self, tar_dim, src_dim, em_dim, num_units, batch_size, **kwargs):
        super(Translator, self).__init__(**kwargs)
        self.optimizer = tf.keras.optimizers.Adam()
        self.encoder = Encoder(src_dim, em_dim, num_units, batch_size)
        self.decoder = Decoder(tar_dim, em_dim, num_units, batch_size)

    def train_step(self, inputs, hidden):
        """implements train_step from Model"""
        inputs, targ = inputs  # split input from Dataset
        # print(inputs.shape, targ.shape)
        loss = 0

        with tf.GradientTape() as tape:
            # pass input into encoder
            enc_output, h, c = self.encoder(inputs, hidden)
            dec_input = targ[:, :-1]  # ignore 0 token
            real = targ[:, 1:]  # ignore <s> token

            # attention mechanism - done
            self.decoder.attention_mechanism.setup_memory(enc_output)

            # initialise AttentionWrapper state as initial state for decoder
            decoder_init_state = self.decoder.build_initial_state(
                self.decoder.batch_sz, [h, c], tf.float32
            )
            # pass input into decoder
            dec_output = self.decoder(dec_input, decoder_init_state)
            dec_output = dec_output.rnn_output
            # targ represent the real values whilst dec_output is a softmax layer
            loss = categorical_loss(real, dec_output)

        var = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, var)

        # apply gradients and return loss
        self.optimizer.apply_gradients(zip(gradients, var))
        # TODO update later to include metrics
        return loss


# init dictionaries
def init_dics():
    """read learned dictionaries for source and target"""
    dic_src.get_stored(os.path.join(cur_dir, "dictionaries", "source_dictionary"))
    dic_tar.get_stored(os.path.join(cur_dir, "dictionaries", "target_dictionary"))


def categorical_loss(real, pred):
    """computes and returns categorical cross entropy"""
    # print(real.shape, pred.shape)

    entropy = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )(real, pred)
    # mask unnecessary symbols
    mask = tf.logical_not(tf.math.equal(real, 0))
    mask = tf.cast(mask, dtype=entropy.dtype)
    entropy *= mask

    # check loss
    # print(tf.keras.backend.get_value(entropy))
    # compute the mean of elements across dimensions of entropy
    return tf.reduce_mean(entropy)


# epochs, batch_size, metrics_rate and cp_rate should be flexible parameters
def train_loop(epochs, data, batch_size, metric_rate, cp_rate):
    """method for RNN train step"""
    # initialise with embedding = 200, units = 200 and batch_size = 200

    model = Translator(len(dic_tar), len(dic_src), 200, 200, batch_size)

    # declare checkpoints
    # set cp directory rrn_checkpoints
    CHECKPOINT_DIR = os.path.join(cur_dir, "rnn_checkpoints")

    for epoch in range(epochs):
        loss = 0
        set_off = time.time()
        hidden = model.encoder.initialize_hidden_state()

        for (i, batch) in enumerate(data):
            # batch loss and epoch avr loss
            b_loss = model.train_step(batch, hidden)
            loss += b_loss

            if i % metric_rate == 0:
                print(
                    "Epoch: {}, Batch: {}, Loss: {:.2f}".format(
                        epoch + 1, i, tf.keras.backend.get_value(b_loss)
                    )
                )

        if (epoch + 1) % cp_rate == 0:
            model.encoder.save_weights(  # saving encoder weights
                os.path.join(
                    CHECKPOINT_DIR,
                    "encoder.epoch{:02d}-loss{:.2f}.hdf5".format(
                        epoch + 1, (tf.keras.backend.get_value(loss) / len(data))
                    ),
                )
            )
            model.decoder.save_weights(  # saving decoder weights
                os.path.join(
                    CHECKPOINT_DIR,
                    "decoder.epoch{:02d}-loss{:.2f}.hdf5".format(
                        epoch + 1, (tf.keras.backend.get_value(loss) / len(data))
                    ),
                )
            )

        # NOTE len(data) produces number of batches in epoch
        print(
            "Epoch: {}, Loss: {:.2f}".format(
                epoch + 1, (tf.keras.backend.get_value(loss) / len(data))
            )
        )
        print("Time taken: {} sec".format(time.time() - set_off))

    return model


def preprocess_data(en_path, de_path):
    """called from main to prepare dataset before initiating training"""
    EPOCHS = 1
    BATCH_SZ = 200
    MET_RATE = 10
    CP_RATE = 1

    # prepare dataset
    data = batches.create_batch_rnn(de_path, en_path)

    tarset = tf.data.Dataset.from_tensor_slices(np.array(data.target))
    data = tf.data.Dataset.from_tensor_slices(np.array(data.source))

    # merge both input points
    data = tf.data.Dataset.zip((data, tarset))
    data = data.shuffle(buffer_size=100).batch(batch_size=BATCH_SZ, drop_remainder=True)
    # data = data.repeat(1)

    # run the train loop
    return (EPOCHS, data, BATCH_SZ, MET_RATE, CP_RATE)


def main():
    """main method"""
    init_dics()
    # encoder = Encoder(len(dic_src), 200, 200, 200)
    # decoder = Decoder(len(dic_tar), 200, 200, 200)

    en_path = os.path.join(cur_dir, "train_data", "multi30k_subword.en")
    de_path = os.path.join(cur_dir, "train_data", "multi30k_subword.de")
    # batch = batches.create_batch_rnn(de_path, en_path)
    epochs, data, sz, met, cp = preprocess_data(en_path, de_path)
    model = train_loop(epochs, data, sz, met, cp)
    rnn_dec.main(model)

    # dataset = tf.data.Dataset(np.array(batch.source))(
    #     batch_size=200, drop_remainder=True
    # )
    # o, h, c = encoder(np.array(batch.source[:200]), encoder.initialize_hidden_state())
    # print(encoder.summary())

    # o = decoder(np.array(batch.target[:200]), [h, c])
    # print(decoder.summary())
    print("\n Result: ok!")  # ok?


if __name__ == "__main__":
    main()
