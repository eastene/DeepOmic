import tensorflow as tf
from tensorflow.python import debug as tf_debug

from model.loss import squared_emphasized_loss

tf.logging.set_verbosity(tf.logging.INFO)
tf.flags.DEFINE_integer("num_parallel_readers", 8, "number of parallel I/O threads")
tf.flags.DEFINE_integer("shuffle_buffer_size", 200, "size (in batches) of in-memory buffer for dataset shuffling")
tf.flags.DEFINE_integer("batch_size", 10, "batch size")
tf.flags.DEFINE_integer("num_parallel_calls", 8, "number of parallel dataset parsing threads "
                                                "(recommended to be equal to number of CPU cores")
tf.flags.DEFINE_integer("prefetch_buffer_size", 200, "size (in batches) of in-memory buffer to prefetch records before parsing")
tf.flags.DEFINE_integer("num_epochs", 100, "number of epochs for training")
tf.flags.DEFINE_string("checkpoint_dir", "/tmp/DeepOmic/", "directory in which to save model checkpoints")

FLAGS = tf.flags.FLAGS

INPUT_FILE_PATTERN = "/home/evan/PycharmProjects/DeepOmic/data/prepared_tf/*.tfrecord"


def omic_data_parse_fn(example):
    # format of each training example
    example_fmt = {
        "X": tf.FixedLenFeature((1317,), tf.float32)  # 1317 = number of SOMA attributes
    }

    parsed = tf.parse_single_example(example, example_fmt)
    return parsed['X']


def input_fn():
    files = tf.data.Dataset.list_files(file_pattern=INPUT_FILE_PATTERN, shuffle=False)

    # interleave reading of dataset for parallel I/O
    dataset = files.apply(
        tf.contrib.data.parallel_interleave(
            tf.data.TFRecordDataset, cycle_length=FLAGS.num_parallel_readers
        )
    )

    dataset = dataset.cache()

    # shuffle data and repeat (if num epochs > 1)
    #dataset = dataset.apply(
    #    tf.contrib.data.shuffle(buffer_size=FLAGS.shuffle_buffer_size)
    #)
    dataset.shuffle(buffer_size=FLAGS.shuffle_buffer_size)

    # parse the data and prepares the batches in parallel (helps most with larger batches)
    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(
            map_func=omic_data_parse_fn, batch_size=FLAGS.batch_size
        )
    )

    # prefetch data so that the CPU can prepare the next batch(s) while the GPU trains
    # recommmend setting buffer size to number of training examples per training step
    dataset = dataset.prefetch(buffer_size=FLAGS.prefetch_buffer_size)

    return dataset


class AutoEncoder:

    def __init__(self, n_hidden, learning_rate):
        # hyper parameters
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate

        # iterator for training examples
        self.dataset = input_fn()
        self.dataset_iter = self.dataset.make_initializable_iterator()
        self.next_elem = self.dataset_iter.get_next()

        # Autoencoder model
        self.x = tf.placeholder(tf.float32, shape=[None, 1317])
        self.encoder = tf.layers.dense(self.x, self.n_hidden, activation=tf.nn.sigmoid)
        self.decoder = tf.layers.dense(self.encoder, self.x.shape[1], activation=tf.nn.sigmoid)
        self.loss = squared_emphasized_loss(labels=self.x, predictions=self.decoder,
                                            corrupted_inds=None, axis=1, alpha=0, beta=1) #tf.losses.cosine_distance(labels=self.x, predictions=self.decoder, axis=1)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())
        self.init_op = tf.global_variables_initializer()

        # Model saver
        self.saver = tf.train.Saver()

    def run_epoch(self):
        pass

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
                c = 0
                n_batches = 0
                sess.run(self.dataset_iter.initializer)

                try:
                    while True:
                        feed_dict = {self.x: sess.run(self.next_elem)}
                        _, c = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                        n_batches += 1

                except tf.errors.OutOfRangeError:
                    pass  # end of data set reached, proceed to next epoch

                if epoch % 3 == 0:
                    print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(c / n_batches))
                    save_path = self.saver.save(sess, FLAGS.checkpoint_dir)
                    print("Model saved in path: %s" % save_path)

    def transform(self, X):
        with tf.Session() as sess:
            X_new = sess.run(self.encoder, {self.x: X})

            return X_new


if __name__ == '__main__':
    ae = AutoEncoder(1000,0.01)
    ae.train()