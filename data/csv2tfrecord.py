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

NUM_CORRUPT_EXAMPLES = 2
CORRUPTION_PR = 0.25  # percent of dimensions to corrupt
CORRUPTION_STR = 1
SEED = None

random.seed(SEED)


def corrupt_random_dimensions(X, skip):
    X_c = X.copy()
    if skip:
        return X_c, np.random.binomial(n=1, p=0, size=X.shape[0]) == 1

    corrupt = np.random.binomial(n=1, p=CORRUPTION_PR, size=X.shape[0]) == 1
    X_c[corrupt] = 0 #X_c[corrupt] + random.uniform(0.01 * CORRUPTION_STR, 1 * CORRUPTION_STR)
    return X_c, corrupt


csv_files = glob(FILE_PATTERN)

for f in csv_files:

    data = pd.read_csv(f, low_memory=False)

    clin, soma, metab = split_omics(data, types=["clinical", "soma", "metab"])

    data = soma
    sids = clin.iloc[:, 0]

    filename = ntpath.basename(f)
    base, ext = os.path.splitext(filename)
    print("Read in {} records".format(data.shape[0]))
    print("Generating {} corrupt records per record read".format(NUM_CORRUPT_EXAMPLES))

    print("Writing " + base + '.tfrecord of {} records'.format(data.shape[0] + (data.shape[0] * NUM_CORRUPT_EXAMPLES)))
    records = 0
    with tf.python_io.TFRecordWriter(base + '.tfrecord') as tfwriter:

        for i in range(data.shape[0]):
            for j in range(NUM_CORRUPT_EXAMPLES + 1):  # do not corrupt on first loop

                X, corrupted_inds = corrupt_random_dimensions(data.iloc[i, :], skip=j == 0)

                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'sid': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(sids[i], encoding='ascii')])),
                            'X': tf.train.Feature(float_list=tf.train.FloatList(value=X)),
                            'Y': tf.train.Feature(float_list=tf.train.FloatList(value=data.iloc[i, :])),
                            # TODO change to ByteList for efficiency
                            'C': tf.train.Feature(int64_list=tf.train.Int64List(value=corrupted_inds)),
                            'is_corr': tf.train.Feature(int64_list=tf.train.Int64List(value=[j != 0])),
                            'FEV1_ch': tf.train.Feature(float_list=tf.train.FloatList(value=[clin['Change_P1_P2_FEV1_ml_yr'][i]])),
                            'Thirona_ch': tf.train.Feature(float_list=tf.train.FloatList(value=[clin['Change_Adj_Density_Thirona'][i]]))
                        }
                    )
                )

                tfwriter.write(example.SerializeToString())
                records += 1

    print("{} records successfully writen".format(records))
