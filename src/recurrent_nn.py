"""
includes the rnn model
"""
import time
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.eager.context import context
from dictionary import dic_tar, dic_src
from utility import cur_dir
import batches
import tensorflow_addons as tfa
import recurrent_dec as rnn_dec


# from decoder import loader, greedy_decoder

# import config file for hyperparameter search space
# import config_custom_train as config

max_line = 0


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()

        # layers of bahdanau attention
        self.l1 = tf.keras.layers.Dense(units, use_bias=False)
        self.l2 = tf.keras.layers.Dense(units, use_bias=False)
        self.attention = tf.keras.layers.AdditiveAttention(dropout=0.2)
        # self.one_dim = tf.keras.layers.Dense(1)

    def call(self, dec_query, enc_values):
        """dec_query: output generated from the lstm layer of decoder
        enc_value: output of the encoder"""
        dec_query_dense = self.l1(dec_query)
        enc_values_dense = self.l2(enc_values)

        # print(enc_values_dense.shape, enc_values.shape, dec_query_dense.shape)
        context_vector, attention_weights = self.attention(
            inputs=[dec_query_dense, enc_values, enc_values_dense],
            return_attention_scores=True,
        )
        # print("done")
        # score = self.one_dim(tf.nn.tanh(self.l1(dec_query) + self.l2(enc_values)))
        # normalise score vector to obtain the attention weights denoted as alpha_i
        # attention_weights = tf.nn.softmax(score, axis=1)
        # context vector is equivalent to the
        # sum_{1 to len(attention_weights)} alpha_i * x_i (encoder output)
        # context_vector = attention_weights * enc_values
        # context_vector = tf.reduce_sum(context_vector, axis=1)
        # context_vector = tf.expand_dims(context_vector, 1)
        # reduce sum sums over all input points
        return context_vector, attention_weights


class Encoder(tf.keras.Model):
    def __init__(self, dic_size, em_dim, num_units, batch_size, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.num_units = num_units
        self.embedding = tf.keras.layers.Embedding(dic_size, em_dim, name="Embedding")

        # The GRU RNN layer processes those vectors sequentially.
        self.gru_forward = tf.keras.layers.GRU(
            self.num_units,
            # Return the sequence and state
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )
        self.gru_backward = tf.keras.layers.GRU(
            self.num_units,
            # Return the sequence and state
            return_sequences=True,
            return_state=True,
            go_backwards=True,
            recurrent_initializer="glorot_uniform",
        )

        self.lstm_forward = tf.keras.layers.LSTM(
            self.num_units,
            return_sequences=True,
            return_state=True,
            name="LSTM_forward",
        )

        self.lstm_backward = tf.keras.layers.LSTM(
            self.num_units,
            return_sequences=True,
            return_state=True,
            go_backwards=True,
            name="LSTM_backword",
        )

        self.bidirect_lstm = tf.keras.layers.Bidirectional(
            self.lstm_forward, backward_layer=self.lstm_backward, merge_mode="sum"
        )
        self.bidirect_gru = tf.keras.layers.Bidirectional(
            self.gru_forward, backward_layer=self.gru_backward, merge_mode="sum"
        )

    def call(self, inputs, hidden=None):
        """implements call from keras.Model"""
        # specify embedding input and pass in embedding in lstm layer
        em = self.embedding(inputs)
        output, _, state = self.bidirect_gru(em, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return [
            [
                tf.zeros((self.batch_size, self.num_units)),
                tf.zeros((self.batch_size, self.num_units)),
            ],
            [
                tf.zeros((self.batch_size, self.num_units)),
                tf.zeros((self.batch_size, self.num_units)),
            ],
        ]


class Decoder(tf.keras.Model):
    def __init__(
        self, vocab_size, embedding_dim, dec_units, batch_sz, attention_type="luong"
    ):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.attention_type = attention_type

        # Embedding Layer to reduce input size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        # Define the fundamental cell for decoder recurrent structure
        self.gru = tf.keras.layers.GRU(
            self.dec_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )

        # Create attention mechanism with memory = None
        self.attention_mechanism = BahdanauAttention(self.dec_units)

        # combine the rnn output and the context vector to generate attention vector
        self.vector = tf.keras.layers.Dense(
            self.dec_units, activation="tanh", use_bias=False
        )

        # Final Dense layer on which softmax will be applied
        self.softmax = tf.keras.layers.Dense(vocab_size, activation="softmax")

    def call(self, inputs, initial_state=None):
        # initial_state should be of size: (batch size, units)
        # send through embedding layer
        inp, enc_output = inputs
        x = self.embedding(inp)

        # process one step with LSTM
        outputs, state = self.gru(x, initial_state=initial_state)

        # use the outputs variable for the attention over encoder's output
        # dec_query: output from decoder's lstm layer, enc_values: output from encoder's lstm layer
        context, weights = self.attention_mechanism(
            dec_query=outputs, enc_values=enc_output
        )

        # concatenate the outputs of lstm and the generated context vector

        rnn_context = tf.concat([context, outputs], axis=-1)
        attention_vector = self.vector(rnn_context)

        # final output layer - applying softmax
        outputs = self.softmax(attention_vector)

        return outputs, weights, state


class Translator(tf.keras.Model):
    def __init__(self, tar_dim, src_dim, em_dim, num_units, batch_size, **kwargs):
        super(Translator, self).__init__(**kwargs)
        self.optimizer = tf.keras.optimizers.Adam()
        self.accuracy = tf.keras.metrics.Accuracy()
        self.encoder = Encoder(src_dim, em_dim, num_units, batch_size)
        self.decoder = Decoder(tar_dim, em_dim, num_units, batch_size)

    def train_step(self, inputs, hidden):
        """implements train_step from Model"""
        inputs, targ = inputs  # split input from Dataset

        # print(inputs.shape, targ.shape)
        loss = 0

        with tf.GradientTape() as tape:
            # pass input into encoder
            # enc_output: whole_sequence_output, h: final_mem_state, c: final_cell_state
            enc_output, state = self.encoder(inputs, None)
            dec_input = targ[:, :-1]  # ignore 0 token
            real = targ[:, 1:]  # ignore <s> token

            # pass input into decoder
            dec_output, weights, state = self.decoder((dec_input, enc_output), None)
            # print(real.shape, dec_output.shape)

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
    """computes and returns sparse categorical cross entropy"""
    # print(real.shape, pred.shape)

    entropy = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False, reduction="none"
    )(real, pred)
    # mask unnecessary symbols
    mask = tf.logical_not(tf.math.equal(real, 0))
    mask = tf.cast(mask, dtype=entropy.dtype)
    entropy *= mask

    # check loss
    # print(tf.keras.backend.get_value(entropy))
    # compute the mean of elements across dimensions of entropy
    return tf.reduce_mean(entropy)


def perplexity(loss):
    """computes and returns perplexity"""
    return tf.keras.backend.exp(loss)


def accuracy():
    """computes and returns accuracy"""
    pass


# epochs, batch_size, metrics_rate and cp_rate should be flexible parameters
def train_loop(epochs, data, batch_size, metric_rate, cp_rate):
    """method for RNN train step"""
    # initialise with embedding = 200, units = 200 and batch_size = 200

    # TODO distinguish between load model or init model und change the check point naming conventions!!

    # model = Translator(len(dic_tar), len(dic_src), 200, 200, batch_size)
    temp = tf.zeros((batch_size, 47))
    model, _, _ = rnn_dec.roll_out_encoder(None, False, batch_size)
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
                val = tf.keras.backend.get_value(b_loss)
                per = perplexity(val)

                print(
                    "Epoch: {}, Batch: {}, Loss: {:.2f}, Per: {:.2f}".format(
                        epoch + 1, i, val, per
                    )
                )

        # saving checkpoints
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
    EPOCHS = 9
    BATCH_SZ = 200
    MET_RATE = 30
    CP_RATE = 3
    # prepare dataset
    global max_line
    max_line, data = batches.create_batch_rnn(de_path, en_path)

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

    # dataset = tf.data.Dataset(np.array(batch.source))(
    #     batch_size=200, drop_remainder=True
    # )
    # o, h, c = encoder(np.array(batch.source[:200]), encoder.initialize_hidden_state())
    # print(encoder.summary())

    # o = decoder(np.array(batch.target[:200]), [h, c])
    # print(decoder.summary())
    print("\n Result: ok!")  # ok?


def test():
    init_dics()
    # encoder = Encoder(len(dic_src), 200, 200, 200)
    # decoder = Decoder(len(dic_tar), 200, 200, 200)

    en_path = os.path.join(cur_dir, "train_data", "multi30k_subword.en")
    de_path = os.path.join(cur_dir, "train_data", "multi30k_subword.de")
    # batch = batches.create_batch_rnn(de_path, en_path)
    epochs, data, sz, met, cp = preprocess_data(en_path, de_path)
    print(max_line)


if __name__ == "__main__":
    main()
    # test()
