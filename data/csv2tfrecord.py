import tensorflow as tf
import pandas as pd
import numpy as np
import random
import ntpath
import os

from glob import glob
from utils.data_utils import split_omics

DATA_DIR = '/home/evan/PycharmProjects/DeepOmic/data/'
FILE_PATTERN = DATA_DIR + '*.csv'

NUM_CORRUPT_EXAMPLES=3
CORRUPTION_PR=0.01
CORRUPTION_STR=1
SEED=None

random.seed(SEED)


def corrupt_random_dimensions(X, skip):
    if skip:
        return X

    return np.apply_along_axis(
        lambda x: x + (0 if random.random() < CORRUPTION_PR else random.uniform(0, 1 * CORRUPTION_STR)),
        axis=0,
        arr=X
    )


csv_files = glob(FILE_PATTERN)


for f in csv_files:

    data = pd.read_csv(f, low_memory=False)

    clin, soma, metab = split_omics(data, types=["clinical", "soma", "metab"])

    filename = ntpath.basename(f)
    base, ext = os.path.splitext(filename)
    print("Read in {} records".format(soma.shape[0]))
    print("Generating {} corrupt records per record read".format(NUM_CORRUPT_EXAMPLES))

    print("Writing " + base + '.tfrecord of {} records'.format(soma.shape[0] * NUM_CORRUPT_EXAMPLES))
    with tf.python_io.TFRecordWriter(base + '.tfrecord') as tfwriter:

        for i in range(soma.shape[0]):
            for j in range(NUM_CORRUPT_EXAMPLES + 1):  # do not corrupt on first loop
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'X': tf.train.Feature(float_list=tf.train.FloatList(
                                value=corrupt_random_dimensions(soma.iloc[i, :], skip=j == 0))
                            ),
                            'Y': tf.train.Feature(float_list=tf.train.FloatList(value=soma.iloc[i, :]))
                        }
                    )
                )

                tfwriter.write(example.SerializeToString())