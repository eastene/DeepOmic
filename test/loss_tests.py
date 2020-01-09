import unittest
import numpy as np
import tensorflow as tf


class LossTestCase(unittest.TestCase):

    def test_squared_emphasized_loss(self):
        from model.loss import squared_emphasized_loss

        alpha = 0.4
        beta = 0.6
        axis = 1

        labels = np.array([[1.0,2.0,3.0], [2.0,3.0,4.0], [4.0,5.0,6.0]])
        predictions = np.array([[0.0, 2.0, 3.0], [3.0, 1.0, 4.0], [6.0, 4.0, 7.0]])
        corrupted_inds = [0]
        uncorrupted_inds = np.delete(np.arange(labels.shape[axis]), corrupted_inds)

        x_c = np.take(labels, corrupted_inds, axis)
        z_c = np.take(predictions, corrupted_inds, axis)
        x = np.take(labels, uncorrupted_inds, axis)
        z = np.take(predictions, uncorrupted_inds, axis)

        expected_output = alpha * np.float_power(x_c - z_c, 2).sum() + beta * np.float_power(x - z, 2).sum()

        labels = tf.convert_to_tensor(labels)
        predictions = tf.convert_to_tensor(predictions)
        loss = squared_emphasized_loss(labels, predictions, corrupted_inds,
                                                axis=axis, alpha=alpha, beta=beta)
        with tf.Session() as sess:
            actual_output = sess.run(loss)
            self.assertAlmostEqual(expected_output, actual_output, places=4)

    def test_cross_entropy_emphasized_loss(self):
        from model.loss import cross_entropy_emphasized_loss

        alpha = 0.4
        beta = 0.6
        axis=1

        labels = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        predictions = np.array([[0.2, 0.1, 0.5], [0.1, 0.5, 0.7], [0.6, 0.4, 0.7]])
        corrupted_inds = [0]
        uncorrupted_inds = np.delete(np.arange(labels.shape[axis]), corrupted_inds)

        x_c = np.take(labels, corrupted_inds, axis)
        z_c = np.take(predictions, corrupted_inds, axis)
        x = np.take(labels, uncorrupted_inds, axis)
        z = np.take(predictions, uncorrupted_inds, axis)

        expected_output = alpha * (-(x_c*np.log(z_c) + (1-x_c)*np.log(1 - z_c)).sum()) + beta * (-(x*np.log(z) + (1-x)*np.log(1 - z)).sum())

        labels = tf.convert_to_tensor(labels)
        predictions = tf.convert_to_tensor(predictions)
        loss = cross_entropy_emphasized_loss(labels, predictions, corrupted_inds,
                                       axis=axis, alpha=alpha, beta=beta)
        with tf.Session() as sess:
            actual_output = sess.run(loss)
            self.assertAlmostEqual(expected_output, actual_output, places=4)