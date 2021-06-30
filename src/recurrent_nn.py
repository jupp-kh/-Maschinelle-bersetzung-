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

# from decoder import loader, greedy_decoder

# import config file for hyperparameter search space
# import config_custom_train as config


class Encoder(tf.keras.Model):
    def __init__(self, dic_size, em_dim, num_units, batch_size, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.num_units = num_units
        self.embedding = tf.keras.layers.Embedding(dic_size, em_dim, name="Embedding")
        self.lstm = tf.keras.layers.LSTM(
            num_units,
            activation="sigmoid",
            return_state=True,
            return_sequences=True,
            name="LSTM",
        )

    def call(self, em, hidden):
        """implements call from keras.Model"""
        # specify embedding input and pass in embedding in lstm layer
        em = self.embedding(em)

        output, h, c = self.lstm(em, initial_state=hidden)
        return output, h, c

    def initialize_hidden_state(self):
        return [
            tf.zeros((self.batch_size, self.num_units)),
            tf.zeros((self.batch_size, self.num_units)),
        ]


class Decoder(tf.keras.Model):
    def __init__(self, dic_size, em_dim, num_units, batch_size, **kwargs):
        """implements init from keras.Model"""
        super(Decoder, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.num_units = num_units

        # input layer and lstm layer
        self.embedding = tf.keras.layers.Embedding(dic_size, em_dim, name="Embedding")
        self.lstm = tf.keras.layers.LSTM(
            num_units,
            activation="sigmoid",
            return_state=True,
            return_sequences=True,
            name="LSTM",
        )

        # tfa required
        # attention layer
        self.attention = num_units

        self.connection = tf.keras.layers.Dense(
            num_units, activation="sigmoid", name="ConnectionLayer"
        )
        # output layer
        self.softmax = Dense(dic_size, activation="softmax", name="Softmax")

    def call(self, inputs, enc_output):
        """implements call from keras.Model"""
        em = self.embedding(inputs)
        mask = self.embedding.compute_mask(inputs)
        output, _, _ = self.lstm(em, initial_state=enc_output, mask=mask)
        flat = self.connection(output)
        return self.softmax(flat)


class Translator(tf.keras.Model):
    def __init__(self, tar_dim, src_dim, em_dim, num_units, batch_size, **kwargs):
        super(Translator, self).__init__(**kwargs)
        self.encoder = Encoder(src_dim, em_dim, num_units, batch_size)
        self.decoder = Decoder(tar_dim, em_dim, num_units, batch_size)

    def train_step(self, inputs, hidden):
        """implements train_step from Model"""
        inputs, targ = inputs  # split input from Dataset
        # print(inputs.shape, targ.shape)
        loss = 0

        with tf.GradientTape() as tape:
            # pass input into encoder
            _, h, c = self.encoder(inputs, hidden)

            # TODO later attention mechanism

            # pass input into decoder
            dec_output = self.decoder(targ, [h, c])
            loss = categorical_loss(targ, dec_output)

        var = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, var)

        # apply gradients and return loss
        self.optimizer.apply_gradients(zip(gradients, var))
        return loss  # update later to include metrics


# init dictionaries
def init_dics():
    """read learned dictionaries for source and target"""
    dic_src.get_stored(os.path.join(cur_dir, "dictionaries", "source_dictionary"))
    dic_tar.get_stored(os.path.join(cur_dir, "dictionaries", "target_dictionary"))


def categorical_loss(real, pred):
    """computes and returns categorical cross entropy"""
    # print(real.shape, pred.shape)
    entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(
        real, pred
    )
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
    model.compile(optimizer="adam")

    # declare checkpoints
    # set cp directory rrn_checkpoints
    CHECKPOINT_DIR = os.path.join(cur_dir, "rnn_checkpoints")

    for epoch in range(epochs):
        loss = 0
        set_off = time.time()
        hidden = model.encoder.initialize_hidden_state()

        for (i, batch) in enumerate(data):
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


def preprocess_data(en_path, de_path):
    """called from main to prepare dataset before initiating training"""
    EPOCHS = 1
    BATCH_SZ = 200
    MET_RATE = 2
    CP_RATE = 1

    # prepare dataset
    data = batches.create_batch_rnn(de_path, en_path)
    tarset = tf.data.Dataset.from_tensor_slices(np.array(data.target))
    data = tf.data.Dataset.from_tensor_slices(np.array(data.source))

    # merge both input points
    data = tf.data.Dataset.zip((data, tarset))
    data = data.shuffle(buffer_size=100).batch(batch_size=BATCH_SZ, drop_remainder=True)
    data = data.repeat(EPOCHS)

    # run the train loop
    train_loop(EPOCHS, data, BATCH_SZ, MET_RATE, CP_RATE)


def main():
    """main method"""
    init_dics()
    # encoder = Encoder(len(dic_src), 200, 200, 200)
    # decoder = Decoder(len(dic_tar), 200, 200, 200)

    en_path = os.path.join(cur_dir, "train_data", "min_train.en")
    de_path = os.path.join(cur_dir, "train_data", "min_train.de")
    # batch = batches.create_batch_rnn(de_path, en_path)
    preprocess_data(en_path, de_path)
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
