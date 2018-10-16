"""
#
# Defines flags used by tensorflow components
#
"""

import tensorflow as tf
from os import path

tf.logging.set_verbosity(tf.logging.ERROR)
tf.flags.DEFINE_integer("num_parallel_readers", 8, "number of parallel I/O threads")
tf.flags.DEFINE_integer("shuffle_buffer_size", 200, "size (in batches) of in-memory buffer for dataset shuffling")
tf.flags.DEFINE_integer("batch_size", 30, "batch size")
tf.flags.DEFINE_integer("num_parallel_calls", 8, "number of parallel dataset parsing threads "
                                                "(recommended to be equal to number of CPU cores")
tf.flags.DEFINE_integer("prefetch_buffer_size", 200, "size (in batches) of in-memory buffer to prefetch records before parsing")
tf.flags.DEFINE_integer("num_comb_epochs", 200, "Number of epochs to train in combination.")
tf.flags.DEFINE_integer("num_epochs", 100, "number of epochs for training")
tf.flags.DEFINE_string("data_dir", path.realpath(__file__), "directory in which input data is located")
tf.flags.DEFINE_string("output_dir", path.realpath(__file__), "directory in which to save encoded data")
tf.flags.DEFINE_string("checkpoint_dir", path.join(path.dirname(path.realpath(__file__)), "tmp", ""),
                       "directory in which to save model checkpoints (by default creates a tmp directory in this file's directory")
tf.flags.DEFINE_float("sparsity_lambda", 0.001, "sparsity constraint on loss")
tf.flags.DEFINE_float("emphasis_alpha", 0.5, "weight given to learning corrupted dimensions")
tf.flags.DEFINE_float("emphasis_beta", 0.5, "weight given to learning uncorrupted dimensions")
tf.flags.DEFINE_integer("num_corrupt", 0, "number of corrupted examples per original examples")
tf.flags.DEFINE_string("output_pattern", "ae_out", "output file pattern for both logging and encoding")
tf.flags.DEFINE_bool("redirect_stdout", False, "redirects anything printed to stdout to file prefixed by output_pattern in output_dir")
tf.flags.DEFINE_float("learn_rate", 0.00001, "learning rate")
tf.flags.DEFINE_integer("input_dims", 1317, "number of dimensions in the input dataset")
tf.flags.DEFINE_list("layers", [1000, 500, 100, 50], "layer sizes from layer 0 to layer N, comma seperated values (e.g. 100,50,10")
FLAGS = tf.flags.FLAGS