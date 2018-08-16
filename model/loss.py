"""
#
# Defines custom loss functions for corrupted dimensions
# (e.g. adding noise to some dimensions)
#
"""

from math import log
import numpy as np
import tensorflow as tf


def squared_emphasized_loss(labels,
                            predictions,
                            corrupted_inds,
                            axis,
                            alpha=0.3,
                            beta=0.7):
    """
    Compute squared loss over training examples that have been
    corrupted along certain dimensions
    :param labels: tensor of training example with no corruption added
    :param predictions: output tensor of autoencoder
    :param corrupted_inds: indices of corrupted dimensions (if any)
    :param axis: axis along which components are taken
    :param alpha: weight for error on components that were corrupted
    :param beta: weight for error on components that were not corrupted
    :return: squared loss, emphasized by corrupted component weight
    """
    assert(labels.shape[axis] == predictions.shape[axis])
    assert(labels.dtype == predictions.dtype)

    uncorrupt = np.delete(np.arange(labels.shape[axis].value), corrupted_inds)
    a = tf.constant(alpha, dtype=labels.dtype)
    b = tf.constant(beta, dtype=labels.dtype)
    x_c = tf.gather(labels, corrupted_inds, axis=axis)
    z_c = tf.gather(predictions, corrupted_inds, axis=axis)
    x = tf.gather(labels, uncorrupt, axis=axis)
    z = tf.gather(predictions, uncorrupt, axis=axis)

    return tf.add(tf.multiply(a, tf.pow(tf.reduce_sum(tf.subtract(x_c, z_c)), 2)),
                  tf.multiply(b, tf.pow(tf.reduce_sum(tf.subtract(x, z)), 2)))


def cross_entropy_emphasized_loss(labels,
                                  predictions,
                                  corrupted_inds,
                                  axis,
                                  alpha=0.3,
                                  beta=0.7):
    """
        Compute cross entropy loss over training examples that have been
        corrupted along certain dimensions
        :param labels: tensor of training example with no corruption added
        :param predictions: output tensor of autoencoder
        :param corrupted_inds: indices of corrupted dimensions (if any)
        :param axis: axis along which components are taken
        :param alpha: weight for error on components that were corrupted
        :param beta: weight for error on components that were not corrupted
        :return: cross entropy loss, emphasized by corrupted component weight
        """
    assert (labels.shape[axis] == predictions.shape[axis])
    assert (labels.dtype == predictions.dtype)

    uncorrupt = np.delete(np.arange(labels.shape[axis].value), corrupted_inds)
    a = tf.constant(alpha, dtype=labels.dtype)
    b = tf.constant(beta, dtype=labels.dtype)
    x_c = tf.gather(labels, corrupted_inds, axis=axis)
    z_c = tf.gather(predictions, corrupted_inds, axis=axis)
    x = tf.gather(labels, uncorrupt, axis=axis)
    z = tf.gather(predictions, uncorrupt, axis=axis)

    return tf.add(tf.multiply(a, tf.pow(tf.reduce_sum(tf.add(tf.multiply(x_c, tf.log(z_c)),
                                                             tf.multiply(1 - x_c, tf.log(1 - z_c)))), 2)),
                  tf.multiply(b, tf.pow(tf.reduce_sum(tf.add(tf.multiply(x, tf.log(z)),
                                                             tf.multiply(1 - x, tf.log(1 - z)))), 2)))