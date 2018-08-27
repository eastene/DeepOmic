import tensorflow as tf
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
        self.masking = tf.layers.dropout(self.input,rate=0.2)

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
        #self.loss = squared_emphasized_loss(labels=self.input, predictions=self.decode_layers[-1].layer,corrupted_inds=None, axis=1, alpha=0, beta=1)
        #self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        #self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        #self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())

        """
        EVAL
        """


        """
        SAVE & RESTORE
        """
        #self.init_op = tf.global_variables_initializer()

    def _initialize_layers(self, input_data_size, layer_list = None):
        prev_output_size = input_data_size
        for ind,layer_size in enumerate(layer_list):
            # Create an Encoder and Decoder for each element in the layer list.
            # For encoders, output size is the current element of layer list.
            # For decoders, output size is equal to the *input* size of the Encoder at this layer.
            # because Decoder(Encoder(data)) == data
            self.encode_layers.append(Encoder(layer_size, name="Encoder_Layer_" + str(ind)))
            self.decode_layers.append(Decoder(prev_output_size, name="Decoder_Layer_" + str(ind)))
            prev_output_size = layer_size

    def make_stack(self, max_lvl = None):
        """
        Make a Encoder/Decoder stack of the type:
        Dec_0(Dec_...(Dec_max_level(Enc_max_level(Enc_...(Enc_0(data))))))

        Accepts:
        max_level - The maximum stack height [0,max_level]

        """

        # Output starts as the placeholder variable for the data.
        output = self.input

        # Set maximum if none specified
        if max_lvl is None:
            max_lvl = len(self.encode_layers)-1

        # Encode step
        for i in range(max_lvl+1):
            output = self.encode_layers[i](output)

        # Decode step
        for i in reversed(range(max_lvl+1)):
            output = self.decode_layers[i](output)

        return output


    def run_epoch(self, sess, train_op, loss):
        c_tot = 0
        n_batches = 0
        sess.run(self.dataset.initialize_train())
        try:
            while True:
                feed_dict={self.input : sess.run(self.next_train_elem)}
                _, c = sess.run([train_op, loss], feed_dict=feed_dict)
                c_tot += c
                n_batches += 1

        except tf.errors.OutOfRangeError:
            return c_tot, n_batches  # end of data set reached, proceed to next epoch
    def train_in_layers(self):
        with tf.Session() as sess:
            # Train each layer separately.
            # First layer trains:
            #  dec_0(enc_0(data))
            #
            # Second layer trains:
            #  dec_1(enc_1(enc_0(data))
            # with enc_0's weights being held constant (not trained), etc
            num_layers = len(self.encode_layers)
            for i in range(1):
                print("Training layer %d out of %d" % (i+1, num_layers))

                network = self.make_stack(i)
                loss = squared_emphasized_loss(labels=self.input,predictions=network,
                                               corrupted_inds=None, axis=1, alpha=0, beta=1)
                optimizer = tf.train.AdamOptimizer(self.learning_rate)

                # Get the highest level decoder and encoder variables, because these are the only ones we want to train.

                train_vars = [var for var in tf.global_variables() if var.name.startswith("Encoder_Layer_" + str(i))
                              or var.name.startswith("Decoder_Layer_" + str(i))]

                train_op=optimizer.minimize(loss,var_list=train_vars)
                init_op = tf.global_variables_initializer()
                saver = tf.train.Saver(var_list=train_vars)
                sess.run(init_op)
                for epoch in range(FLAGS.num_epochs):

                    self.run_epoch(sess,train_op,loss)
                    print("Epoch done.")


    def train_full(self):
        """Trains all layers of the Autoencoder Stack at once."""
        with tf.Session() as sess:

            writer = tf.summary.FileWriter('.')

            network = self.make_stack()

            loss = squared_emphasized_loss(labels=self.input, predictions=network,
                                                corrupted_inds=None, axis=1, alpha=0, beta=1)
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            train_op = optimizer.minimize(loss)
            init_op = tf.global_variables_initializer()
            saver = tf.train.Saver()
            #TODO remove after debugging
            #sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type="readline")  # readline for PyCharm interface
            if tf.train.checkpoint_exists(FLAGS.checkpoint_dir):
                saver.restore(sess, FLAGS.checkpoint_dir)
                print("Model restored.")
            else:
                sess.run(init_op)

            for epoch in range(FLAGS.num_epochs):

                c, n_batches = self.run_epoch(sess, train_op=train_op,loss=loss)

                print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(c / n_batches))
                save_path = saver.save(sess, FLAGS.checkpoint_dir)
                print("Model saved in path: %s" % save_path)

            # evaluate
            m_tot = 0
            n_batches = 0
            sess.run(self.dataset.initialize_eval())
            self.distance = tf.square(tf.subtract(self.input, network))
            self.eval_op = tf.reduce_sum(self.distance)
            try:
                while True:
                    feed_dict = {self.input: sess.run(self.next_eval_elem)}
                    m = sess.run([self.eval_op], feed_dict=feed_dict)
                    m_tot += m[0]
                    n_batches += 1

            except tf.errors.OutOfRangeError:
                pass

            print("Training Accuracy: {}".format(m_tot / n_batches))


if __name__ == '__main__':
    dom = DeepOmicModel(0.0001)
    dom.train_full()