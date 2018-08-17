"""
#
# Defines input pipeline using tensorflow Dataset API
#
"""

import tensorflow as tf

from model.flags import FLAGS

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

    # shuffle data
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
