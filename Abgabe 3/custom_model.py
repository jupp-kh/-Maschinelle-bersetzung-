"""
Contains class ExtModel implement Model
Contains implementation of keras.callbacks.Callback: ExtCallback 
Contains metric class Perplexity

* Contains network archictecture for word embedding: Modell
"""
import tensorflow as tf
from tensorflow.keras.layers import Input, Concatenate, Embedding, Dense
from tensorflow.keras.models import Model
from dictionary import dic_tar, dic_src

from tensorflow.python.ops.variables import trainable_variables


# globals sit here


# code begins here


class Perplexity(tf.keras.metrics.Metric):
    def __init__(self, name="perplexity", **kwargs):
        super(Perplexity, self).__init__(name=name, **kwargs)
        self.cross_entropy = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        self.perplexity = self.add_weight(name="tp", initializer="zeros")

    # used for loading models
    def get_config(self):
        """Returns the serializable config of the metric."""
        # config = {"cross_entropy": self.cross_entropy, "perplexity": self.perplexity}
        base_config = super(Perplexity, self).get_config()
        return base_config

    def _calculate_perplexity(self, y_true, y_pred):
        """
        Method returns perplexity
        """
        perplexity = tf.keras.backend.exp(self.cross_entropy(y_true, y_pred))
        return perplexity

    def update_state(self, y_true, y_pred, sample_weight=0):
        if sample_weight is not None:
            self.log(
                self.WARNING,
                "Provided 'sample_weight' argument to the perplexity metric. "
                "Currently this is not handled and won't do anything differently.",
            )

        # Remember self.perplexity is a tensor (tf.Variable),
        # FIXME: there is an issue with updating perplexity using this class :(
        self.perplexity = self._calculate_perplexity(y_true, y_pred)

    def result(self):
        return self.perplexity

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.perplexity = tf.Variable(0)


# using this callback the perplexity gets updated properly
class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, display):
        self.seen = 0
        self.display = display

    # call back seems to update perplexity at a better rate
    def on_batch_end(self, batch, logs):
        self.seen += 1
        if self.seen % self.display == 0:
            outlog = logs
            outlog["perplexity"] = float(tf.exp(logs["loss"]).numpy())
            for key in outlog:
                outlog[key] = "{:.2f}".format(outlog[key])
            print("After", self.seen, "batches:", outlog)


# class declaration
class WordLabelerModel(Model):
    # initialise model and build layers immediatly.
    def __init__(self, window_size=2):
        #     # building model
        input_point_1 = Input(shape=(2 * window_size + 1,), name="I0")

        x = Embedding(len(dic_src), 250)(input_point_1)
        x = Dense(units=200, activation="relu", name="FullyConSource")(x)

        input_point_2 = Input(shape=(window_size,), name="I1")
        y = Embedding(len(dic_tar), 250, name="EmbeddedTarget")(input_point_2)
        y = Dense(units=200, activation="relu", name="FullyConTarget")(y)

        fully_concat = Concatenate(axis=1, name="ConcatLayer")([x, y])

        fully_connected_one = Dense(
            200, activation="relu", name="FullyConnectedLayer1"
        )(fully_concat)

        # second fully connected layer / projection
        fully_connected_two = Dense(
            200, activation=None, input_shape=(len(dic_tar), 1)
        )(fully_connected_one)

        fully_connected_two = tf.keras.layers.Flatten(name="Flatten")(
            fully_connected_two
        )
        # softmax layer
        softmax_layer = Dense(len(dic_tar), activation="softmax", name="Softmax")(
            fully_connected_two
        )

        # in1 = in_src.extend(in_tar)
        # build model and specify input and output points.
        # call super()
        super(WordLabelerModel, self).__init__(
            inputs=[input_point_1, input_point_2], outputs=[softmax_layer]
        )

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


# class FeedForward:
#     """
#     Class for our feed forward network
#     """

#     def __init__(self):
#         self.model = None

#     # the model
#     def build_model(self, w=2):
#         """
#         build our neural network model
#         """
#         # integrate_gpu()

#         input_point_1 = Input(shape=(2 * w + 1,), name="I0")

#         x = Embedding(len(dic_src), 250)(input_point_1)
#         x = Dense(units=200, activation="relu", name="FullyConSource")(x)

#         input_point_2 = Input(shape=(w,), name="I1")
#         y = Embedding(len(dic_tar), 250, name="EmbeddedTarget")(input_point_2)
#         y = Dense(units=200, activation="relu", name="FullyConTarget")(y)

#         fully_concat = Concatenate(axis=1, name="ConcatLayer")([x, y])

#         fully_connected_one = Dense(
#             200, activation="relu", name="FullyConnectedLayer1"
#         )(fully_concat)

#         # second fully connected layer / projection
#         fully_connected_two = Dense(
#             200, activation=None, input_shape=(len(dic_tar), 1)
#         )(fully_connected_one)

#         fully_connected_two = tf.keras.layers.Flatten(name="Flatten")(
#             fully_connected_two
#         )
#         # softmax layer
#         softmax_layer = Dense(len(dic_tar), activation="softmax", name="Softmax")(
#             fully_connected_two
#         )

#         # in1 = in_src.extend(in_tar)
#         # build final model
#         self.model = ExtModel(
#             inputs=[input_point_1, input_point_2], outputs=[softmax_layer]
#         )

#     def compile_model(self):
#         """
#         compiles model
#         """
#         from tensorflow.keras.optimizers import Adam

#         self.model.compile(
#             optimizer=Adam(),
#             loss="categorical_crossentropy",
#             # using categorical cross entropy from keras provided one-hot vectors
#             metrics=[
#                 "accuracy",
#                 Perplexity(),
#             ],
#         )

#     def show_summary(self):
#         print(self.model.summary())