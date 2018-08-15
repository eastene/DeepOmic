import tensorflow as tf


tf.logging.set_verbosity(tf.logging.INFO)
tf.flags.DEFINE_integer("num_parallel_readers", 1, "number of parallel I/O threads")
tf.flags.DEFINE_integer("shuffle_buffer_size", 3, "size (in batches) of in-memory buffer for dataset shuffling")
tf.flags.DEFINE_integer("batch_size", 10, "batch size")
tf.flags.DEFINE_integer("num_parallel_calls", 1, "number of parallel dataset parsing threads "
                                                "(recommended to be equal to number of CPU cores")
tf.flags.DEFINE_integer("prefetch_buffer_size", 3, "size (in batches) of in-memory buffer to prefetch records before parsing")
tf.flags.DEFINE_integer("num_epochs", 100, "number of epochs for training")

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
    dataset = dataset.apply(
        tf.contrib.data.shuffle_and_repeat(buffer_size=FLAGS.shuffle_buffer_size)
    )

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

    def __init__(self, n_visible, n_hidden, learning_rate, n_epochs=10, batch_size=100):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate

        # iterator for training examples
        self.dataset = input_fn()
        self.dataset_iter = self.dataset.make_initializable_iterator()
        self.next_elem = self.dataset_iter.get_next()

        # Autoencoder model
        self.x = self.next_elem  # tf.placeholder(tf.float32, shape=[1, 1317])
        self.encoder = tf.layers.dense(self.x, self.n_visible, activation=tf.nn.sigmoid)
        self.decoder = tf.layers.dense(self.encoder, self.n_hidden, activation=tf.nn.sigmoid)
        self.loss = tf.losses.cosine_distance(labels=self.x, predictions=self.decoder, axis=1)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())
        self.init_op = tf.global_variables_initializer()

    def run_epoch(self):
        pass

    def train(self):
        """
        # iterator for training examples
        dataset = input_fn()
        dataset_iter = dataset.make_initializable_iterator()
        next_elem = dataset_iter.get_next()

        # Autoencoder model
        x = next_elem #tf.placeholder(tf.float32, shape=[1, 1317])
        encoder = tf.layers.dense(x, self.n_visible, activation=tf.nn.sigmoid)
        decoder = tf.layers.dense(encoder, self.n_hidden, activation=tf.nn.sigmoid)
        loss = tf.losses.cosine_distance(labels=x, predictions=decoder, axis=1)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        init_op = tf.global_variables_initializer()
        """

        #TODO Reenable GPU
        with tf.Session() as sess:
            sess.run(self.init_op)
            sess.run(self.dataset_iter.initializer)

            for epoch in range(FLAGS.num_epochs):
                avg_cost = 0.0
                #TODO make batches = num_examples // batch_size
                for batch in range(108):

                    _, c = sess.run([self.train_op, self.loss])

                    # TODO make batches = num_examples // batch_size
                    avg_cost += c / 108

                    if epoch % 3 == 0 and batch == 0:
                        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))


if __name__ == '__main__':
    ae = AutoEncoder(100,1317,0.01)
    ae.train()