"""
#
# Defines input pipeline using tensorflow Dataset API
#
"""

import tensorflow as tf
from os import path
import random

from model.flags import FLAGS


class InputPipeline:

    def __init__(self, file_pattern, size_of_split=5):
        """
        Input pipeline based on the Tensorflow Dataset API
        :param file_pattern: regex pattern of files to include as input (.tfrecord)
        :param size_of_split: number of batches to hold for evaluation
        :param num_corrupt_examples: number of corrupted examples to produce for each example read
        :param corruption_pr: probability of corruption occuring for each dimension values between (0-1)
        :param corruption_str: strength of corruption, multiplies corruption value applied to corrupt dimensions
        :param seed: seed value for corruption value, default uses system time
        """
        self.data_dir = FLAGS.data_dir
        self.file_pattern = file_pattern
        self.search_pattern = path.join(self.data_dir, self.file_pattern)
        self.dataset = self.input_fn()
        self.eval_iter = self.dataset.take(size_of_split).make_initializable_iterator()
        self.train_iter = self.dataset.skip(size_of_split).make_initializable_iterator()

    def omic_data_parse_fn(self, example):
        # format of each training example
        example_fmt = {
            "sid": tf.FixedLenFeature((), tf.string),
            "X": tf.FixedLenFeature((FLAGS.input_dims,), tf.float32),  # 1317 = number of SOMA attributes
            "Y": tf.FixedLenFeature((FLAGS.input_dims,), tf.float32),  # 1317 = number of SOMA attributes
            "C": tf.FixedLenFeature((FLAGS.input_dims,), tf.int64),
            "is_corr": tf.FixedLenFeature((1,), tf.int64),
            "FEV1_ch": tf.FixedLenFeature((1,), tf.float32),
            "Thirona_ch": tf.FixedLenFeature((1,), tf.float32)
        }

        parsed = tf.parse_single_example(example, example_fmt)
        sid = tf.cast(parsed['sid'], dtype=tf.string)
        C = tf.cast(parsed['C'], dtype=tf.bool)
        is_corr = tf.cast(parsed['is_corr'], dtype=tf.bool)

        return sid, parsed['X'], C, is_corr, parsed['Y'], parsed["FEV1_ch"], parsed["Thirona_ch"]

    def input_fn(self):
        print("Looking for data files matching: {}\nIn: {}".format(self.file_pattern, self.data_dir))
        files = tf.data.Dataset.list_files(file_pattern=self.search_pattern, shuffle=False)

        # interleave reading of dataset for parallel I/O
        dataset = files.apply(
            tf.contrib.data.parallel_interleave(
                tf.data.TFRecordDataset, cycle_length=FLAGS.num_parallel_readers
            )
        )

        #dataset = dataset.cache()

        # shuffle data
        dataset.shuffle(buffer_size=FLAGS.shuffle_buffer_size)

        # parse the data and prepares the batches in parallel (helps most with larger batches)
        dataset = dataset.apply(
            tf.contrib.data.map_and_batch(
                map_func=self.omic_data_parse_fn, batch_size=FLAGS.batch_size
            )
        )

        # prefetch data so that the CPU can prepare the next batch(s) while the GPU trains
        # recommmend setting buffer size to number of training examples per training step
        dataset = dataset.prefetch(buffer_size=FLAGS.prefetch_buffer_size)

        return dataset

    def initialize_train(self):
        return self.train_iter.initializer

    def initialize_eval(self):
        return self.eval_iter.initializer

    def next_train_elem(self):
        return self.train_iter.get_next()

    def next_eval_elem(self):
        return self.eval_iter.get_next()