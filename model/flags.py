"""
#
# Defines flags used by tensorflow components
#
"""

import tensorflow as tf
from os import path

tf.logging.set_verbosity(tf.logging.ERROR)

# Input flags
tf.flags.DEFINE_string("data_dir", path.split(path.realpath(__file__))[0], "directory in which input data is located")
tf.flags.DEFINE_string("input_pattern", "*.tfrecord", "pattern of input files")

# Output flags
tf.flags.DEFINE_string("output_pattern", "ae_out", "output file pattern for both logging and encoding")
tf.flags.DEFINE_bool("no_timestamp", False, "do not add timestamp to output filename if set")
tf.flags.DEFINE_bool("plot_2D", False, "produce a 2D PC plot of the embeddings after encoding")
tf.flags.DEFINE_string("output_dir", path.split(path.realpath(__file__))[0], "directory in which to save encoded data")
tf.flags.DEFINE_bool("redirect_stdout", False, "redirects anything printed to stdout to a file prefixed by "
                                               "output_pattern in output_dir")

# Model sizing and checkpointing flags
tf.flags.DEFINE_integer("input_dims", 1317, "number of dimensions in the input dataset")
tf.flags.DEFINE_list("layers", [1000, 500, 100, 50], "layer sizes from layer 0 to layer N, comma seperated values "
                                                     "(e.g. 100,50,10")
tf.flags.DEFINE_string("checkpoint_dir", path.join(path.dirname(path.realpath(__file__)), "tmp", ""),
                       "directory in which to save model checkpoints (by default creates a tmp directory in "
                       "this file's directory")

# Training flags
tf.flags.DEFINE_integer("batch_size", 32, "batch size")
tf.flags.DEFINE_integer("num_epochs", 100, "number of epochs for pre-training")
tf.flags.DEFINE_integer("num_comb_epochs", 200, "Number of epochs to train in combination (finetuning).")
tf.flags.DEFINE_float("learn_rate", 0.01, "learning rate")
tf.flags.DEFINE_integer("num_corrupt", 0, "number of corrupted examples per original examples")

# Loss function flags
tf.flags.DEFINE_float("sparsity_lambda", 0, "sparsity constraint on loss")
tf.flags.DEFINE_float("emphasis_alpha", 1, "weight given to learning corrupted dimensions")
tf.flags.DEFINE_float("emphasis_beta", 1, "weight given to learning uncorrupted dimensions")
tf.flags.DEFINE_enum("reg", None, ["l1", "l2"], "regularization applied to layer weights during pre-training")

# Input pipeline flags (can leave at defaults in most cases)
tf.flags.DEFINE_integer("num_parallel_readers", 8, "number of parallel I/O threads")
tf.flags.DEFINE_integer("shuffle_buffer_size", 200, "size (in batches) of in-memory buffer for dataset shuffling")
tf.flags.DEFINE_integer("num_parallel_calls", 8, "number of parallel dataset parsing threads "
                                                 "(recommended to be equal to number of CPU cores")
tf.flags.DEFINE_integer("prefetch_buffer_size", 200, "size (in batches) of in-memory buffer to prefetch "
                                                     "records before parsing")

# Reproducability flags
tf.flags.DEFINE_integer("seed", 1234, "seed for initialization of layers")

FLAGS = tf.flags.FLAGS