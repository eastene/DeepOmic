import unittest
import numpy as np
import tensorflow as tf

class LossTestCase(unittest.TestCase):

    def test_squared_emphasized_loss(self):
        from model.loss import squared_emphasized_loss

        labels = np.array([[1.0,2.0,3.0], [2.0,3.0,4.0], [4.0,5.0,6.0]])
        labels = tf.convert_to_tensor(labels)
        predictions = np.array([[0.0, 2.0, 3.0], [3.0, 1.0, 4.0], [6.0, 4.0, 7.0]])
        predictions = tf.convert_to_tensor(predictions)
        corrupted_ind = [0]

        expected_output = 4.0
        loss = squared_emphasized_loss(labels, predictions, corrupted_ind,
                                                axis=1, alpha=0.4, beta=0.6)
        with tf.Session() as sess:
            actual_output = sess.run(loss)
            self.assertAlmostEqual(expected_output, actual_output, places=4)