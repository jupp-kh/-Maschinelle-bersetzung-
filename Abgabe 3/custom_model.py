"""
Creates Class ExtModel 
"""
import tensorflow as tf
import numpy as np


from tensorflow.python.ops.variables import trainable_variables


# globals sit here


# code begins here

# use this rather than calling sparse_categorical_crossentropy
def crossentropy(labels, logits):
    """
    computes loss function with sparse categorical crossentropy
    """
    return tf.keras.losses.categorical_crossentropy(labels, logits, from_logits=False)


def perplexity(labels, logits):
    """
    computes and returns perplexity for given labels and logits
    """
    cross_entropy = crossentropy(labels, logits)
    perplexity = tf.keras.backend.exp(cross_entropy)
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

        # print(x)
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

        # update metrics (including specified metrics in Mode.compile() as well)
        self.compiled_metrics.update_state(y, y_pred)

        # return a dictionary mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
