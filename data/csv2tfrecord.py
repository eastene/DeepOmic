import tensorflow as tf
import pandas as pd
import ntpath
import os

from glob import glob
from utils.data_utils import split_omics

DATA_DIR = '/home/evan/PycharmProjects/DeepOmic/data/'
FILE_PATTERN = DATA_DIR + '*.csv'

csv_files = glob(FILE_PATTERN)

for f in csv_files:

    data = pd.read_csv(f, low_memory=False)

    clin, soma, metab = split_omics(data, types=["clinical", "soma", "metab"])

    filename = ntpath.basename(f)
    base, ext = os.path.splitext(filename)
    print("Writing " + base + '.tfrecord')

    with tf.python_io.TFRecordWriter(base + '.tfrecord') as tfwriter:

        for i in range(soma.shape[0]):
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'X': tf.train.Feature(float_list=tf.train.FloatList(value=soma.iloc[i, :])),
                    }
                )
            )

            tfwriter.write(example.SerializeToString())