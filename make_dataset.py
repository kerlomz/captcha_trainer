#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import sys
import random
import tensorflow as tf
from config import *
from constants import RunMode

_RANDOM_SEED = 0
label_max_length = 0

TFRECORDS_TYPE = [
    RunMode.Trains,
    RunMode.Test
]

if not os.path.exists(TFRECORDS_DIR):
    os.makedirs(TFRECORDS_DIR)


def _image(path):
    with open(path, "rb") as f:
        return f.read()


def _dataset_exists(dataset_dir):
    for split_name in TFRECORDS_TYPE:
        output_filename = os.path.join(dataset_dir, "{}_{}.tfrecords".format(TARGET_MODEL, split_name.value))
        if not tf.gfile.Exists(output_filename):
            return False
    return True


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def image_to_tfrecords(image_data, label):
    return tf.train.Example(features=tf.train.Features(feature={
        'image': bytes_feature(image_data),
        'label': bytes_feature(label),
    }))


def _convert_dataset(file_list, mode):
    output_filename = os.path.join(TFRECORDS_DIR, "{}_{}.tfrecords".format(TARGET_MODEL, mode.value))
    with tf.python_io.TFRecordWriter(output_filename) as writer:
        for i, file_name in enumerate(file_list):
            try:
                sys.stdout.write('\r>> Converting image %d/%d ' % (i + 1, len(file_list)))
                sys.stdout.flush()
                image_data = _image(file_name)
                labels = re.search(TRAINS_REGEX, file_name.split(PATH_SPLIT)[-1])
                if labels:
                    labels = labels.group()
                else:
                    raise NameError('invalid filename {}'.format(file_name))
                labels = labels.encode('utf-8')

                example = image_to_tfrecords(image_data, labels)
                writer.write(example.SerializeToString())

            except IOError as e:
                print('could not read:', file_list[1])
                print('error:', e)
                print('skip it \n')
    sys.stdout.write('\n')
    sys.stdout.flush()


def make_dataset():
    dataset_path = DATASET_PATH
    if _dataset_exists(TFRECORDS_DIR):
        print('Exists!')
    else:
        if not DATASET_PATH and isinstance(TRAINS_PATH, str) and not TRAINS_PATH.endswith("tfrecords"):
            dataset_path = TRAINS_PATH
        elif not DATASET_PATH and isinstance(TRAINS_PATH, str) and TRAINS_PATH.endswith("tfrecords"):
            print('DATASET_PATH is not configured!')
            exit(-1)

        if isinstance(dataset_path, list):
            origin_dataset = []
            for trains_path in dataset_path:
                origin_dataset += [os.path.join(trains_path, trains) for trains in os.listdir(trains_path)]
        else:
            origin_dataset = [os.path.join(TRAINS_PATH, trains) for trains in os.listdir(dataset_path)]

        random.seed(_RANDOM_SEED)
        random.shuffle(origin_dataset)
        test_dataset = origin_dataset[:TEST_SET_NUM]
        trains_dataset = origin_dataset[TEST_SET_NUM:]

        _convert_dataset(test_dataset, mode=RunMode.Test)
        _convert_dataset(trains_dataset, mode=RunMode.Trains)
        print("Done!")


if __name__ == '__main__':
    make_dataset()
