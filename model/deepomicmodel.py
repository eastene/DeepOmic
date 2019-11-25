from datetime import datetime
from time import time
import os.path as path
from os import makedirs

import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

from model.decoder import Decoder
from model.encoder import Encoder
from model.input_pipeline import InputPipeline
from model.loss import *
from model.utils import print_config, redirects_stdout

FILE_PATTERN = "*.tfrecord"


class DeepOmicModel:

    def __init__(self):
        """
        DeepOmicModel

        Autoencoder for exploring parameterization of different autoencoder features
        (e.g. Sparse Autoencoder, Denoising Autoencoder, etc.) and applying them
        to high dimensional omic's data.
        """

        """
        HYPER PARAMETERS
        """
        # Loss HPs
        self.alpha = FLAGS.emphasis_alpha
        self.beta = FLAGS.emphasis_beta
        self.lam = FLAGS.sparsity_lambda
        self.loss = mean_squared_error  # squared_emphasized_sparse_loss
        self.optimizer = tf.train.AdamOptimizer

        # Model HPs
        self.layers = FLAGS.layers
        self.learning_rate = FLAGS.learn_rate
        self.input_dims = FLAGS.input_dims

        """
        INPUT PIPELINE
        """

        # Dataset Iterators
        self.dataset = InputPipeline(FILE_PATTERN)
        self.next_train_elem = self.dataset.next_train_elem()
        self.next_eval_elem = self.dataset.next_eval_elem()
        self.next_encode_elem = self.dataset.next_encode_elem()

        """
        MODEL COMPONENTS
        """
        # Model Inputs
        self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dims])
        self.corrupt_mask = tf.placeholder(dtype=tf.bool, shape=[None, self.input_dims])
        self.expected = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dims])
        self.is_corr = tf.placeholder(dtype=tf.bool)
        self.predictor = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        # List of autoencoder layers
        self.encode_layers = []
        self.decode_layers = []
        # List of tuples containing the size of each layer
        self.layer_sizes = []
        # List of Saver objects for each layer. Each layer has a separate Saver so we can restore the layers
        # independently of each other.
        self.layer_savers = []
        # Prefixes for naming the encoder and decoder tensors, as well as the checkpoint directories.
        self.encoder_prefix = "Encoder_Layer_"
        self.decoder_prefix = "Decoder_Layer_"

        """
        MODEL INITIALIZATION
        """
        tf.set_random_seed(FLAGS.seed)
        self._initialize_layers(self.input_dims, self.layers)

    def get_loss_func(self, labels, predictions, encoded, is_corr, lam, corrupted_inds, axis, alpha, beta,
                      ignore_corr=False, regularizer=None):
        return self.loss(labels=labels, predictions=predictions, encoded=encoded, is_corr=is_corr,
                         corrupted_inds=corrupted_inds, lam=lam, axis=axis, alpha=alpha, beta=beta,
                         ignore_corr=ignore_corr, regularizer=regularizer)

    def get_optimizer(self, learning_rate, name="adam"):
        return self.optimizer(learning_rate=learning_rate, name=name)

    def get_enc_dec_name(self, num):
        return self.encoder_prefix + str(num), self.decoder_prefix + str(num)

    def get_layer_checkpoint_dirname(self, num):
        in_size, out_size = self.layer_sizes[num]
        return self.get_enc_dec_name(num)[0] + "_" + str(in_size) + "_" + str(out_size) + "/"

    def _initialize_layers(self, input_data_size, layer_list=None):
        prev_output_size = input_data_size
        for ind, layer_size in enumerate(layer_list):
            # Create an Encoder and Decoder for each element in the layer list.
            # For encoders, output size is the current element of layer list.
            # For decoders, output size is equal to the *input* size of the Encoder at this layer.
            # because Decoder(Encoder(data)) == data
            self.encode_layers.append(Encoder(layer_size, name="Encoder_Layer_" + str(ind)))
            if ind == 0:
                self.decode_layers.append(Decoder(prev_output_size, name="Decoder_Layer_" + str(ind), end=True))
            else:
                self.decode_layers.append(Decoder(prev_output_size, name="Decoder_Layer_" + str(ind)))

            self.layer_sizes.append((prev_output_size, layer_size))
            # Build checkpoint directories for each layer.
            cpt_dirname = FLAGS.checkpoint_dir + self.get_layer_checkpoint_dirname(ind)

            if not path.isdir(cpt_dirname):
                makedirs(cpt_dirname)

            prev_output_size = layer_size

    def make_stack(self, max_lvl=None):
        """
        Make a Encoder/Decoder stack of the type:
        Dec_0(Dec_...(Dec_max_level(Enc_max_level(Enc_...(Enc_0(data))))))

        Accepts:
        max_level - The maximum stack height [0,max_level]
        """

        # Output starts as the placeholder variable for the data.
        output = tf.identity(self.input)

        # Set maximum if none specified
        if max_lvl is None:
            max_lvl = len(self.encode_layers) - 1

        # Encode step
        for i in range(max_lvl + 1):
            output = self.encode_layers[i](tf.identity(output))

        # Decode step
        for i in reversed(range(max_lvl + 1)):
            output = self.decode_layers[i](tf.identity(output), self.encode_layers[i].kernel)

        return output

    def run_epoch(self, sess, train_op, loss, use_clin_feat=False):
        c_tot = 0
        n_batches = 0
        sess.run(self.dataset.initialize_train())
        try:
            while True:
                sids, x, cm, is_corr, y = sess.run(self.next_train_elem)
                # train on X, and corruptions of X
                feed_dict = {
                    self.input: x,
                    self.corrupt_mask: cm,
                    self.is_corr: is_corr,
                    self.expected: y
                }

                if use_clin_feat:
                    feed_dict = {
                        self.input: x,
                        self.corrupt_mask: cm,
                        self.is_corr: is_corr,
                        self.expected: y
                    }

                # Run encode operation
                _, c = sess.run([train_op, loss], feed_dict=feed_dict)

                c_tot += c
                n_batches += 1

        except tf.errors.OutOfRangeError:
            return c_tot, n_batches  # end of data set reached, proceed to next epoch

    def train_in_layers(self, start_layer=0):
        with tf.Session() as sess:
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

            # Train each layer separately.
            # First layer trains:
            #  dec_0(enc_0(data))
            #
            # Second layer trains:
            #  dec_0(dec_1(enc_1(enc_0(data)))
            # with enc_0 and dec_0's weights being held constant, etc
            num_layers = len(self.encode_layers)
            # Train layers individually
            for i in range(start_layer, num_layers):
                print("Training layer %d out of %d" % (i + 1, num_layers))

                # Build the network stack for this depth.
                network = self.make_stack(i)

                # Retrieve the prefix names of the encoder and decoder at this level
                encoder_pref, decoder_pref = self.get_enc_dec_name(i)

                # Get the loss function specified and pass it the "clean" input
                regularizer = None if i == num_layers-1 else tf.nn.l2_loss
                loss = self.get_loss_func(labels=self.expected, predictions=network,
                                          encoded=self.encode_layers[i].output, is_corr=self.is_corr,
                                          corrupted_inds=self.corrupt_mask, lam=self.lam, axis=0, alpha=self.alpha,
                                          beta=self.beta, regularizer=regularizer)

                # for testing, use *pure* mean squared error
                loss_sme = self.get_loss_func(labels=self.expected, predictions=network,
                                              encoded=self.encode_layers[i].output, is_corr=self.is_corr,
                                              lam=0, corrupted_inds=self.corrupt_mask, axis=0, alpha=1,
                                              beta=1, ignore_corr=True)

                # Get the specified optimizer.
                optimizer = self.get_optimizer(self.learning_rate, "adam_layers")

                # Get the variables that will be trained by the optimizer (this should be only the variables for
                # this level's encoder and decoder
                train_vars = [var for var in tf.global_variables() if var.name.startswith(encoder_pref)
                              or var.name.startswith(decoder_pref)]

                #  Tell optimizer to minimize only the variables at this level.
                self.train_op = optimizer.minimize(loss, var_list=train_vars)

                # Run initializer operation. Only initialize variables from the current layer.
                sess.run(tf.global_variables_initializer())

                # Create a new layer saver for this layer.
                self.layer_savers.append(tf.train.Saver(train_vars, name="Level_" + str(i) + "_Saver"))

                # If possible, restore the variables from a checkpoint at this level.
                for j in range(i + 1):
                    if tf.train.checkpoint_exists(FLAGS.checkpoint_dir + self.get_layer_checkpoint_dirname(j)):
                        print("Restoring layer " + str(j + 1) + " from checkpoint.")
                        self.layer_savers[j].restore(sess, FLAGS.checkpoint_dir + self.get_layer_checkpoint_dirname(j))
                    else:
                        print("No previous checkpoint found for layer " + str(j + 1) + ", layer will be initialized.")

                # Iterate through FLAGS.num_epochs epochs for each layer of training.
                # writer = tf.summary.FileWriter(".")
                # writer.add_graph(tf.get_default_graph())
                for epoch in range(FLAGS.num_epochs):
                    # Run the epoch
                    c, n_batches = self.run_epoch(sess, self.train_op, loss)
                    # Save the result in a checkpoint directory with the layer name.
                    self.layer_savers[i].save(sess, FLAGS.checkpoint_dir + self.get_layer_checkpoint_dirname(i))
                    ts_loss, ts_loss_pure = self.get_test_acc(sess, loss, loss_sme)
                    print("\rLoss: {:.3f}".format(c / n_batches) + " Test Set Loss: {:.3f}".format(
                        ts_loss) + " SME-only Test Loss: {:.3f}".format(ts_loss_pure) + " at Epoch " + str(epoch),
                          end="")
                print("\n")

            # Train layers together
            print("Training network in combination for %d epochs." % FLAGS.num_comb_epochs)
            # Full depth network
            network = self.make_stack()
            # Optimize all variables at once
            optimizer = self.get_optimizer(self.learning_rate / 100, "adam_comb")
            regularizer = None  # tf.nn.l2_loss
            # Get the loss function specified and pass it the "clean" input
            loss = self.get_loss_func(labels=self.expected, predictions=network, encoded=self.encode_layers[-1].output,
                                      is_corr=self.is_corr, lam=self.lam, corrupted_inds=self.corrupt_mask, axis=0,
                                      alpha=self.alpha, beta=self.beta, regularizer=regularizer)
            # for testing, use *pure* mean squared error
            loss_sme = self.get_loss_func(labels=self.expected, predictions=network,
                                          encoded=self.encode_layers[-1].output, is_corr=self.is_corr,
                                          lam=0, corrupted_inds=self.corrupt_mask, axis=0, alpha=1,
                                          beta=1, ignore_corr=True)

            self.train_op = optimizer.minimize(loss)

            # Initialize variables.
            sess.run(tf.global_variables_initializer())
            # Restore layers
            for j in range(num_layers):
                if tf.train.checkpoint_exists(FLAGS.checkpoint_dir + self.get_layer_checkpoint_dirname(j)):
                    print("Restoring layer " + str(j + 1) + " from checkpoint.")
                    self.layer_savers[j].restore(sess, FLAGS.checkpoint_dir + self.get_layer_checkpoint_dirname(j))
                else:
                    print("ERROR: Layer %d not found." % j + 1)
            # Run for FLAGS.num_comb_epochs epochs
            for i in range(FLAGS.num_comb_epochs):
                c, n_batches = self.run_epoch(sess, self.train_op, loss)
                # print("Saving %d layers.\n" % num_layers)
                for j in range(num_layers):
                    self.layer_savers[j].save(sess, FLAGS.checkpoint_dir + self.get_layer_checkpoint_dirname(j))
                ts_loss, ts_loss_pure = self.get_test_acc(sess, loss, loss_sme)
                print("\rLoss: {:.3f}".format(c / n_batches) + " Test Set Loss: {:.3f}".format(
                    ts_loss) + " SME-only Test Loss: {:.3f}".format(ts_loss_pure) + " at Epoch " + str(i), end="")
            print("\n")

            # Test regression performance

    def get_test_acc(self, sess, loss, sme_loss, use_clin_feat=False):
        # evaluate
        m_tot = 0
        sme_tot = 0
        n_batches = 0
        sess.run(self.dataset.initialize_eval())
        # self.distance = tf.square(tf.subtract(self.input, network))
        # self.eval_op = tf.reduce_mean(self.distance)
        while True:
            try:
                # train on X, and corruptions of X
                sids, x, cm, is_corr, y = sess.run(self.next_eval_elem)
                feed_dict = {self.input: x,
                             self.corrupt_mask: cm,
                             self.is_corr: is_corr,
                             self.expected: y}
                if use_clin_feat:
                    feed_dict = {
                        self.input: x,
                        self.corrupt_mask: cm,
                        self.is_corr: is_corr,
                        self.expected: y
                    }
                m, m_sme = sess.run([loss, sme_loss], feed_dict=feed_dict)
                m_tot += m
                sme_tot += m_sme
                n_batches += 1
            except tf.errors.OutOfRangeError:
                break
        return m_tot / n_batches, sme_tot / n_batches

    def regression_select(self):
        num_layers = len(self.encode_layers)
        with tf.Session() as sess:
            # Train layers together
            print("Training network with regression for %d epochs." % FLAGS.num_comb_epochs)
            # Full depth network
            network = self.make_stack()
            # Optimize all variables at once
            optimizer = self.get_optimizer(self.learning_rate / 100, "adam_comb_regression")

            initializer = tf.random_uniform_initializer(minval=-0.01, maxval=0.01)
            kernel = tf.get_variable("kernel", [FLAGS.layers[-1], 1], initializer=initializer,
                                     dtype=tf.float32)
            bias = tf.get_variable("bias", [1], initializer=initializer, dtype=tf.float32)
            # self.output = self.activation(tf.matmul(input,self.kernel) + self.bias)
            regressor = tf.matmul(self.encode_layers[-1].output, kernel) + bias

            # for testing, use *pure* mean squared error
            loss = tf.losses.mean_squared_error(labels=self.predictor, predictions=regressor)

            train_op = optimizer.minimize(loss)

            # Initialize variables.
            sess.run(tf.global_variables_initializer())
            # Restore layers
            for j in range(num_layers):
                if tf.train.checkpoint_exists(FLAGS.checkpoint_dir + self.get_layer_checkpoint_dirname(j)):
                    print("Restoring layer " + str(j + 1) + " from checkpoint.")
                    self.layer_savers[j].restore(sess, FLAGS.checkpoint_dir + self.get_layer_checkpoint_dirname(j))
                else:
                    print("ERROR: Layer %d not found." % j + 1)
            # Run for FLAGS.num_comb_epochs epochs
            for i in range(FLAGS.num_comb_epochs):
                c, n_batches = self.run_epoch(sess, train_op, loss)
                # print("Saving %d layers.\n" % num_layers)
                for j in range(num_layers):
                    self.layer_savers[j].save(sess, FLAGS.checkpoint_dir + self.get_layer_checkpoint_dirname(j))
                ts_loss, ts_loss_pure = self.get_test_acc(sess, loss, loss)
                print("\rLoss: {:.3f}".format(c / n_batches) + " Test Set Loss: {:.3f}".format(
                    ts_loss) + " SME-only Test Loss: {:.3f}".format(ts_loss_pure) + " at Epoch " + str(i), end="")
            print("\n")

    def encode(self, to_file="", est_sil=False):
        # output dataframe
        df = pd.DataFrame(columns=["EM{}".format(i) for i in range(int(self.layers[-1]))])
        df.index.rename("sid", inplace=True)

        with tf.Session() as sess:
            # Restore layers
            for j in range(len(self.encode_layers)):
                if tf.train.checkpoint_exists(FLAGS.checkpoint_dir + self.get_layer_checkpoint_dirname(j)):
                    print("Restoring layer " + str(j + 1) + " from checkpoint.")
                    self.layer_savers[j].restore(sess, FLAGS.checkpoint_dir + self.get_layer_checkpoint_dirname(j))
                else:
                    print("ERROR: Layer %d not found." % (j + 1))
                    exit(1)

            # prep data iterator
            sess.run(self.dataset.initialize_encode())

            # encode dataset
            while True:
                try:
                    sids, x, cm, is_corr, y = sess.run(self.next_encode_elem)

                    # remove any corrupted rows from encoding
                    x = x[~is_corr[:, 0], :]

                    # do not encode empty input
                    if x.size == 0:
                        continue

                    # encode X
                    feed_dict = {
                        self.input: x,
                        self.corrupt_mask: cm,
                        self.expected: y
                    }

                    # Run encode operation
                    out = sess.run(self.encode_layers[-1].output, feed_dict=feed_dict)

                    for row in range(out.shape[0]):
                        df.loc[sids[row].decode('ascii')] = out[row]

                except tf.errors.OutOfRangeError:
                    break

        df = df.sort_index()

        if to_file:
            df.to_csv(path.join(path.dirname(FLAGS.output_dir), to_file))

        scores_km = []
        scores_ac = []
        if est_sil:
            for i in range(2, 10):
                km = KMeans(n_clusters=i)
                ac = AgglomerativeClustering(n_clusters=i)
                labels_km = km.fit_predict(df)
                labels_ac = ac.fit_predict(df)
                scores_km.append(silhouette_score(df, labels_km))
                scores_ac.append(silhouette_score(df, labels_ac))
                print("KM CLUSTERS : {}, SIL_SCORE : {}".format(i, scores_km[-1]))
                print("AG CLUSTERS : {}, SIL_SCORE : {}".format(i, scores_ac[-1]))

        return df.values

    def plot_results(self):
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt

        # TODO: Remove
        #labs = np.squeeze(pd.read_csv('soma_labels.csv').astype(np.float32).values)

        data = self.encode()
        pca = PCA(n_components=2)
        xy = pca.fit_transform(data[:, :])
        plt.scatter(xy[:, 0], xy[:, 1]) #c=labs)
        plt.title("alpha={} beta={} lambda={} ncorr={}".format(
            FLAGS.emphasis_alpha,
            FLAGS.emphasis_beta,
            FLAGS.sparsity_lambda,
            FLAGS.num_corrupt
        ))
        plt.show()


@redirects_stdout
def runner():
    print("Training Started at: {}".format(datetime.now()))
    timestamp = str(time()).replace('.', '')
    print("Assigning this run UID: {}".format(timestamp))
    print_config()
    dom = DeepOmicModel()
    dom.train_in_layers()
    # dom.regression_select()
    dom.encode("{}_{}.csv".format(FLAGS.output_pattern, timestamp), True)
    dom.plot_results()
    print("Training Ended at: {}".format(datetime.now()))


if __name__ == '__main__':
    runner()
