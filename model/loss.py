"""
#
# Defines custom loss functions for corrupted dimensions
# (e.g. adding noise to some dimensions)
#
"""

import numpy as np
import tensorflow as tf

from model.flags import FLAGS


def squared_emphasized_loss(labels,
                            predictions,
                            corrupted_inds=None,
                            axis=0,
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

    num_elems = labels.shape[axis].value * FLAGS.batch_size

    # corrupted features
    x_c = tf.boolean_mask(labels, corrupted_inds)
    z_c = tf.boolean_mask(predictions, corrupted_inds)
    # uncorrupted features
    x = tf.boolean_mask(labels, ~corrupted_inds)
    z = tf.boolean_mask(predictions, ~corrupted_inds)

    # if training on examples with corrupted indices
    if x_c is not None:
        lhs = alpha * tf.reduce_sum(tf.square(tf.subtract(x_c, z_c)))
        rhs = beta * tf.reduce_sum(tf.square(tf.subtract(x, z)))

    # if training on uncorrupted examples, no need to select indices and alpha effectively 0
    else:
        lhs = 0.0
        rhs = 1.0 * tf.reduce_sum(tf.square(tf.subtract(labels, predictions)))

    return tf.add(lhs, rhs) / num_elems


def cross_entropy_emphasized_loss(labels,
                                  predictions,
                                  corrupted_inds,
                                  axis=0,
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

    num_elems = labels.shape[axis].value * FLAGS.batch_size

    # corrupted features
    x_c = tf.boolean_mask(labels, corrupted_inds)
    z_c = tf.boolean_mask(predictions, corrupted_inds)
    # uncorrupted features
    x = tf.boolean_mask(labels, ~corrupted_inds)
    z = tf.boolean_mask(predictions, ~corrupted_inds)

    # if training on examples with corrupted indices
    if x_c is not None:
        lhs = alpha * (-tf.reduce_sum(tf.add(tf.multiply(x_c, tf.log(z_c)),
                                             tf.multiply(1.0 - x_c, tf.log(1.0 - z_c)))))
        rhs = beta * (-tf.reduce_sum(tf.add(tf.multiply(x, tf.log(z)),
                                             tf.multiply(1.0 - x, tf.log(1.0 - z)))))
    else:
        lhs = 0
        rhs = -tf.reduce_sum(tf.add(tf.multiply(labels, tf.log(predictions)),
                                             tf.multiply(1.0 - labels, tf.log(1.0 - predictions))))

    return tf.add(lhs, rhs) / num_elems
