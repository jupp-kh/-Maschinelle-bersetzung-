from metrics import met_bleu
from keras.layers import CuDNNLSTM, Input, Concatenate, Embedding, Dense, Bidirectional
from tensorflow.keras.models import Model
import tensorflow as tf
from dictionary import dic_tar, dic_src
import os
from utility import cur_dir
import sys
import decoder
import glob

# from decoder import loader, greedy_decoder

from tensorflow.python.ops.variables import trainable_variables

# import config file for hyperparameter search space
import config_custom_train as config


class Recurrent_model(Model):
    # initialise model and build layers immediatly.
    def __init__(self, **kwargs):

        encoder_input = Input(shape=(None,))

        encoder_embedded = Embedding(input_dim=len(dic_tar), output_dim=100)(
            encoder_input
        )
        output, state_h, state_c = Bidirectional(CuDNNLSTM(200, return_sequences=True))(
            encoder_embedded
        )
        encoder_state = [state_h, state_c]

        decoder_input = layers.Input(shape=(None,))
decoder_embedded = layers.Embedding(input_dim=decoder_vocab, output_dim=64)(
    decoder_input
)



        super(Recurrent_model, self).__init__()

    def train_step(self, data):
        """
        Implements train_step from tf.keras.Model
        Provides specialised functionality and an additional
        static variables for saving certain informational
        attributes required by users
        """
        # split dataset and convert y to a one_hot representative
        x, y = data
        ### This line is necessary
        y = tf.one_hot(y, depth=len(dic_tar))

        # Gradient Tape tracks the automatic differentiation that occurs in a TF model.
        # its context records computations to get the gradient of any tensor computed
        # while recording with regards to any trainable data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            # calc loss value
            # the type of funtion could be set in Model.compile()
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Run backward pass
        # compute gradients
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y, y_pred)

        # Collect metrics to return
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics

    # compiles our defined word model
    def compile_model(self):
        """
        compiles model
        """
        from tensorflow.keras.optimizers import Adam

        self.compile(
            optimizer=Adam(),
            loss="categorical_crossentropy",
            # using categorical cross entropy from keras provided one-hot vectors
            metrics=[
                "accuracy",
                Perplexity(),
            ],
        )

    def test_step(self, data):
        # Unpack the data
        x, y = data
        y = tf.one_hot(y, depth=len(dic_tar))
        # Compute predictions
        y_pred = self(x, training=False)
        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}
