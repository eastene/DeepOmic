import tensorflow as tf
#from tensorflow.python import debug as tf_debug
import numpy as np

from model.flags import FLAGS
from model.loss import squared_emphasized_loss
from model.input_pipeline import InputPipeline

from model.decoder import Decoder
from model.encoder import Encoder

FILE_PATTERN = "*.tfrecord"

class DeepOmicModel:

    def __init__(self, learning_rate):
        self.dataset = InputPipeline(FILE_PATTERN)
        self.next_train_elem = self.dataset.next_train_elem()
        self.next_eval_elem = self.dataset.next_eval_elem()

        self.input = tf.placeholder(dtype=tf.float32, shape=[None, 1317])
        self.corrupt_mask = tf.placeholder(dtype=tf.bool, shape=[None, 1317])
        self.expected = tf.placeholder(dtype=tf.float32, shape=[None, 1317])

        """
        MODEL
        """
        # List of autoencoder layers
        self.encode_layers = []
        self.decode_layers = []
        self._initialize_layers(1317, [1000,500,250])

        """
        TRAIN
        """
        self.learning_rate = learning_rate
        print(str(self.input.shape) + "  " + str(self.decode_layers[-1].layer.shape))
        self.predictions = self.decode_layers[-1].layer
        # TODO: get corrupted indices to work correctly
        self.loss = squared_emphasized_loss(labels=self.expected, predictions=self.decode_layers[-1].layer,
                                            corrupted_inds=self.corrupt_mask, axis=1, alpha=0.4, beta=0.6)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())

        """
        EVAL
        """
        self.distance = tf.square(tf.subtract(self.input, self.decode_layers[-1].layer))
        self.eval_op = tf.reduce_mean(self.distance)

        """
        SAVE & RESTORE
        """
        self.saver = tf.train.Saver()

        self.init_op = tf.global_variables_initializer()

    def _initialize_layers(self, input_data_size, layer_list = None):
        prev_enc_layer = self.input

        # Build Encoder layers
        for ind,layer_size in enumerate(layer_list):
            dropout=False
            if ind == 0:
                dropout = False
            layer_encoder = Encoder(prev_enc_layer,layer_size,name="Encoder_Layer_"+str(ind),dropout=dropout)
            self.encode_layers.append(layer_encoder)
            prev_enc_layer = layer_encoder.layer

        prev_dec_layer = self.encode_layers[-1]
        # Build Decoder layers
        for ind,enc_layer in reversed(list(enumerate(self.encode_layers))):
            layer_decoder = Decoder(prev_dec_layer.layer,enc_layer.input.shape[1], name="Decoder_Layer_"+str(ind))
            self.decode_layers.append(layer_decoder)
            prev_dec_layer = layer_decoder

    def run_epoch(self, sess):
        c_tot = 0
        n_batches = 0
        sess.run(self.dataset.initialize_train())
        try:
            while True:
                x, cm, y = sess.run(self.next_train_elem)
                # train on X, and corruptions of X
                feed_dict = {
                    self.input: x,
                    self.corrupt_mask: cm,
                    self.expected: y
                }
                _, c, dl = sess.run([self.train_op, self.loss, self.decode_layers[-1].layer], feed_dict=feed_dict)
                c_tot += c
                n_batches += 1

        except tf.errors.OutOfRangeError:
            return c_tot, n_batches  # end of data set reached, proceed to next epoch

    def train(self):
        with tf.Session() as sess:
            self.writer = tf.summary.FileWriter('.')
            self.writer.add_graph(tf.get_default_graph())
            #TODO remove after debugging
            #sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type="readline")  # readline for PyCharm interface

            if tf.train.checkpoint_exists(FLAGS.checkpoint_dir):
                self.saver.restore(sess, FLAGS.checkpoint_dir)
                print("Model restored.")
            else:
                sess.run(self.init_op)

            for epoch in range(FLAGS.num_epochs):

                c, n_batches = self.run_epoch(sess)

                print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(c / n_batches))
                save_path = self.saver.save(sess, FLAGS.checkpoint_dir)
                print("Model saved in path: %s" % save_path)

            # evaluate
            m_tot = 0
            n_batches = 0
            sess.run(self.dataset.initialize_eval())

            try:
                while True:
                    x, cm, y = sess.run(self.next_eval_elem)
                    # train on X, and corruptions of X
                    feed_dict = {
                        self.input: x,
                        self.corrupt_mask: cm,
                        self.expected: y
                    }
                    m = sess.run([self.eval_op], feed_dict=feed_dict)
                    m_tot += m[0]
                    n_batches += 1

            except tf.errors.OutOfRangeError:
                pass

            print("Mean Sum of Squares: {}".format(m_tot / n_batches))

    def transform(self, X):
        # Oper TF Session
        with tf.Session() as sess:
            if tf.train.checkpoint_exists(FLAGS.checkpoint_dir):
                self.saver.restore(sess, FLAGS.checkpoint_dir)
                print("Model loaded.")
            else:
                print("No existing encoder found.")
                exit(1)

            #X_new = sess.run(self.encode_layers[], {self.input: X})
            cur_data = X
            for layer in self.encode_layers:
                cur_data = layer.transform(sess,cur_data)

            return cur_data

    def reverse_transform(self, X):

        with tf.Session() as sess:
            if tf.train.checkpoint_exists(FLAGS.checkpoint_dir):
                self.saver.restore(sess, FLAGS.checkpoint_dir)
                print("Model loaded.")
            else:
                print("No existing decoder found.")
                exit(1)

            cur_data = X
            for layer in self.decode_layers:
                cur_data = layer.reverse_transform(sess, cur_data)

            return cur_data


if __name__ == '__main__':
    dom = DeepOmicModel(0.001)
    dom.train()