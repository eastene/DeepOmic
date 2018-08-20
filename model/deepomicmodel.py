import tensorflow as tf
from model.flags import FLAGS
from model.loss import squared_emphasized_loss
from model.input_pipeline import input_fn
from model.autoencoder import AutoEncoder

class DeepOmicModel:
    def __init__(self, learning_rate):
        self.dataset = input_fn()
        self.dataset_iter = self.dataset.make_initializable_iterator()
        self.next_elem = self.dataset_iter.get_next()
        self.autoencoder = AutoEncoder(1000)

        self.learning_rate = learning_rate
        # Input and output for the autoencoder
        self.input = self.autoencoder.x
        self.output = self.autoencoder.decoder

        self.loss = squared_emphasized_loss(labels=self.input, predictions=self.autoencoder.decoder,corrupted_inds=None, axis=1, alpha=0, beta=1) #tf.losses.cosine_distance(labels=self.x, predictions=self.decoder, axis=1)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())
        self.init_op = tf.global_variables_initializer()

        # Model saver
        self.saver = tf.train.Saver()

    def run_epoch(self, sess):
        c = 0
        n_batches = 0
        sess.run(self.dataset_iter.initializer)

        try:
            while True:
                feed_dict = {self.input: sess.run(self.next_elem)}
                _, c = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                n_batches += 1

        except tf.errors.OutOfRangeError:
            return c, n_batches  # end of data set reached, proceed to next epoch

    def train(self):
        with tf.Session() as sess:
            #TODO remove after debugging
            #sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type="readline")  # readline for PyCharm interface

            if tf.train.checkpoint_exists(FLAGS.checkpoint_dir):
                self.saver.restore(sess, FLAGS.checkpoint_dir)
                print("Model restored.")
            else:
                sess.run(self.init_op)

            for epoch in range(FLAGS.num_epochs):

                c, n_batches = self.run_epoch(sess)

                if epoch % 3 == 0:
                    print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(c / n_batches))
                    save_path = self.saver.save(sess, FLAGS.checkpoint_dir)
                    print("Model saved in path: %s" % save_path)

    def transform(self, X):
        with tf.Session() as sess:
            if tf.train.checkpoint_exists(FLAGS.checkpoint_dir):
                self.saver.restore(sess, FLAGS.checkpoint_dir)
                print("Model loaded.")
            else:
                print("No existing encoder found.")
                exit(1)

            X_new = sess.run(self.autoencoder.encoder, {self.input: X})

            return X_new

    def reverse_transform(self, X):
        assert(X.shape[1] == self.autoencoder.n_hidden)  # assert reversing already transformed data

        with tf.Session() as sess:
            if tf.train.checkpoint_exists(FLAGS.checkpoint_dir):
                self.saver.restore(sess, FLAGS.checkpoint_dir)
                print("Model loaded.")
            else:
                print("No existing decoder found.")
                exit(1)

            X_new = sess.run(self.autoencoder.decoder, {self.input: X})

            return X_new

if __name__ == '__main__':
    dom = DeepOmicModel(0.01)
    dom.train()