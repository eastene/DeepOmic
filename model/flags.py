"""
#
# Defines flags used by tensorflow components
#
"""

import tensorflow as tf

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