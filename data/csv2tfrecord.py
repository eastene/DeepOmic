from glob import glob
import pandas as pd
import numpy as np
import random
import os

import tensorflow as tf
from sklearn.preprocessing import minmax_scale, normalize


def csv2tfrecord(file_pattern="*.csv",
                 data_dir=".",
                 num_corrupt=0,
                 corruption_prct=75,
                 scale=False,
                 normal=False,
                 seed=1234):
    DATA_DIR = data_dir
    FILE_PATTERN = os.path.join(DATA_DIR, file_pattern)

    NUM_CORRUPT_EXAMPLES = num_corrupt  # number of corrupt examples
    CORRUPTION_PR = corruption_prct  # percent of dimensions to corrupt
    SEED = seed
    SCALE = False
    NORMALIZE = False

    random.seed(SEED)
    csv_files = glob(FILE_PATTERN)

    def corrupt_random_dimensions(X, skip):
        X_c = X.copy()
        if skip:
            return X_c, np.zeros_like(X_c, dtype=np.int8)

        corrupt = np.random.binomial(n=1, p=CORRUPTION_PR, size=X.shape[0]) == 1
        X_c[corrupt] = X_c[corrupt] + (
                np.random.choice([-1, 1], size=X_c[corrupt].shape) * np.random.ranf(size=X_c[corrupt].shape) * X_c[
            corrupt])
        return X_c, corrupt

    for f in csv_files:

        data = pd.read_csv(f, low_memory=False, delimiter=',', index_col=0)

        sids = data.index

        if SCALE:
            data = minmax_scale(data)
        elif NORMALIZE:
            data = normalize(data)
        else:
            data = data.values

        base, ext = os.path.splitext(f)
        print("Read in {} records with {} columns".format(data.shape[0], data.shape[1]))
        print("Generating {} corrupt records per record read".format(NUM_CORRUPT_EXAMPLES))

        print("Writing " + base + '.tfrecord of {} records'.format(
            data.shape[0] + (data.shape[0] * NUM_CORRUPT_EXAMPLES)))
        records = 0
        with tf.io.TFRecordWriter(base + '.tfrecord') as tfwriter:

            for i in range(data.shape[0]):
                for j in range(NUM_CORRUPT_EXAMPLES + 1):  # do not corrupt on first loop

                    X, corrupted_inds = corrupt_random_dimensions(data[i, :], skip=j == 0)

                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                'sid': tf.train.Feature(
                                    bytes_list=tf.train.BytesList(value=[bytes(sids[i], encoding='ascii')])),
                                'X': tf.train.Feature(float_list=tf.train.FloatList(value=X)),
                                'Y': tf.train.Feature(float_list=tf.train.FloatList(value=data[i, :])),
                                # TODO change to ByteList for efficiency
                                'C': tf.train.Feature(int64_list=tf.train.Int64List(value=corrupted_inds)),
                                'is_corr': tf.train.Feature(int64_list=tf.train.Int64List(value=[j != 0]))  # ,
                                # 'clin_feat': tf.train.Feature(float_list=tf.train.FloatList(value=[clin[clin_feat][i]]))
                            }
                        )
                    )

                    tfwriter.write(example.SerializeToString())
                    records += 1

        print("{} records successfully writen".format(records))
