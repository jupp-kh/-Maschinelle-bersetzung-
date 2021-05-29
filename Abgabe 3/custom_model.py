"""
Creates Class ExtModel 
"""
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.layers.core import Dense
import batches
from batches import Batch, get_next_batch
from tensorflow.keras.layers import Input, Concatenate, Embedding
from tensorflow.keras.models import Model
from dictionary import dic_tar, dic_src
import os

from tensorflow.python.ops.variables import trainable_variables


# globals sit here


# code begins here


class ExtCallback(tf.keras.callbacks.Callback):
    def __init__(self, display):
        self.seen = 0
        self.display = display

    def on_batch_end(self, batch, logs):
        self.seen += logs.get("size", 0)
        print(logs)
        print("perplexity:", int(tf.exp(logs["loss"]).numpy()))
        if self.seen % self.display:
            print("\n{}\{} - loss .... \n".format(self.seen, logs.keys()))


# use this rather than calling sparse_categorical_crossentropy
def crossentropy(labels, logits):
    """
    computes loss function with sparse categorical crossentropy
    """
    return tf.keras.losses.categorical_crossentropy(labels, logits, from_logits=False)


def perplexity(labels, logits):
    """
    computes and returns perplexity for given labels and logits
     A low perplexity indicates the probability distribution is good at predicting the sample.
    """
    cross_entropy = crossentropy(labels, logits)
    perplexity = tf.exp(cross_entropy)
    return perplexity


# class declaration
class ExtModel(tf.keras.Model):
    def train_step(self, data):
        """
        Implements train_step from tf.keras.Model
        Provides specialised functionality and an additional
        static variables for saving certain informational
        attributes required by users
        """
        # split data into inputs and outputs
        x, y = data
        y = tf.one_hot(y, depth=len(dic_tar))
        print(x)
        print(y)

        # print(tf.keras.backend.get_value(y))
        # Gradient Tape tracks the automatic differentiation that occurs in a TF model.
        # its context records computations to get the gradient of any tensor computed
        # while recording with regards to any trainable data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            # calc loss value
            # the type of funtion could be set in Model.compile()
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # update model's weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # update metrics (including specified metrics in Model.compile() as well)
        # self.reset_state()
        self.compiled_metrics.update_state(y, y_pred)

        # return a dictionary mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


class Modell:
    """
    Class for our feed forward network
    """

    def __init__(self):
        self.model = None

    # the model
    def build_model(self, w=2):
        """
        build our neural network model
        """
        # integrate_gpu()

        input_point_1 = Input(shape=(2 * w + 1,), name="I0")

        x = Embedding(
            len(dic_src),
            100,
        )(input_point_1)
        x = Dense(units=100, activation="relu", name="FullyConSource")(x)

        input_point_2 = Input(shape=(w,), name="I1")
        y = Embedding(len(dic_tar), 100, name="EmbeddedTarget")(input_point_2)
        y = Dense(units=100, activation="relu", name="FullyConTarget")(y)

        fully_concat = Concatenate(axis=1, name="ConcatLayer")([x, y])

        fully_connected_one = Dense(
            100, activation="relu", name="FullyConnectedLayer1"
        )(fully_concat)

        # second fully connected layer / projection
        fully_connected_two = Dense(
            100, activation=None, input_shape=(len(dic_tar), 1)
        )(fully_connected_one)

        fully_connected_two = tf.keras.layers.Flatten(name="Flatten")(
            fully_connected_two
        )
        # softmax layer
        softmax_layer = Dense(len(dic_tar), activation="softmax", name="Softmax")(
            fully_connected_two
        )

        # in1 = in_src.extend(in_tar)
        # build final model
        self.model = ExtModel(
            inputs=[input_point_1, input_point_2], outputs=[softmax_layer]
        )

    def compile_model(self):
        """
        compiles model
        """
        from tensorflow.keras.optimizers import Adam

        self.model.compile(
            optimizer=Adam(),
            loss=crossentropy,
            # using categorical cross entropy from keras provided one-hot vectors
            metrics=[
                "accuracy",
            ],
        )

    def show_summary(self):
        print(self.model.summary())