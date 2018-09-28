"""
#
# Defines simple Autoencoder with encoder/decoder layers
#
"""

import tensorflow as tf
from tensorflow.python import debug as tf_debug

from model.flags import FLAGS
from model.loss import squared_emphasized_loss
from model.input_pipeline import input_fn


class AutoEncoder:

    def __init__(self, n_hidden):
        # hyper parameters
        self.n_hidden = n_hidden

        # Autoencoder model
        self.x = tf.placeholder(tf.float32, shape=[None, 1317])
        # TODO add support for encoding corrupted examples, possibly by adding a new variable to hold them
        self.encoder = tf.layers.dense(self.x, self.n_hidden, activation=tf.nn.sigmoid)
        self.decoder = tf.layers.dense(self.encoder, self.x.shape[1], activation=tf.nn.sigmoid)
