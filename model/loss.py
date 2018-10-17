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


def squared_emphasized_sparse_loss(labels,
                            predictions,
                            encoded,
                            is_corr,
                            corrupted_inds=None,
                            lam=0.01,
                            axis=0,
                            alpha=0.3,
                            beta=0.7):
    """
        Compute squared loss over training examples that have been
        corrupted along certain dimensions and impose sparsity constraint
        :param labels: tensor of training example with no corruption added
        :param predictions: output tensor of autoencoder
        :param encoded: output tensor of encoder layer
        :param corrupted_inds: indices of corrupted dimensions (if any)
        :param lam: lambda penalty term for sparsity constraint (higher = more sparsity)
        :param axis: axis along which components are taken
        :param alpha: weight for error on components that were corrupted
        :param beta: weight for error on components that were not corrupted
        :return: squared loss, emphasized by corrupted component weight
    """
    assert (labels.dtype == predictions.dtype)
    assert (beta + alpha == 1.0)

    # sparsity penalty, added to each sample
    omega = lam * tf.reduce_sum(tf.abs(encoded)) if lam != 0 else 0.0

    # if training on uncorrupted examples, no need to select indices and alpha effectively 0
    if beta == 1:
        uncorr = (~is_corr)
        uncorr.set_shape([None])
        labs_uncorr = tf.boolean_mask(labels, uncorr, axis=axis, name='labels_uncorrupt')
        preds_uncorr = tf.boolean_mask(predictions, uncorr, axis=axis, name='predictions_uncorrupt')
        axis_mean = tf.reduce_mean(tf.square(tf.subtract(labs_uncorr, preds_uncorr)), axis=axis)

    # if training on examples with corrupted indices
    else:
        # Multiply boolean mask by alpha to multiply each value by alpha in the end
        mults = tf.scalar_mul(alpha, tf.cast(corrupted_inds, dtype=tf.float32))
        mults = mults + tf.scalar_mul(beta, tf.cast(~corrupted_inds, dtype=tf.float32))
        squares = tf.square(tf.subtract(labels, predictions))
        axis_mean = tf.reduce_mean(squares * mults, axis=axis)

    return tf.reduce_mean(axis_mean) # + omega)


def squared_sparse_loss(labels,
                            predictions,
                            encoded,
                            lam=0.01,
                            axis=0):
    """
        Compute squared loss over training examples and impose
        sparsity constraint
        :param labels: tensor of training example with no corruption added
        :param predictions: output tensor of autoencoder
        :param encoded: output tensor of encoder layer
        :param lam: lambda penalty term for sparsity constraint (higher = more sparsity)
        :param axis: axis along which components are taken
        :return: squared loss
    """
    assert (labels.shape[axis] == predictions.shape[axis])
    assert (labels.dtype == predictions.dtype)

    num_elems = labels.shape[axis].value * FLAGS.batch_size

    # sparsity penalty, added to each sample
    omega = lam * tf.reduce_sum(tf.abs(encoded))

    loss = tf.reduce_sum(tf.square(tf.subtract(labels, predictions))) + omega

    return loss / num_elems