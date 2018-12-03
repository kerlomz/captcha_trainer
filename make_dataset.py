#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import sys
import random
import tensorflow as tf
from config import *

REGEX_MAP = {
    RunMode.Trains: TRAINS_REGEX,
    RunMode.Test: TEST_REGEX
}

_RANDOM_SEED = 0

if not os.path.exists(TFRECORDS_DIR):
    os.makedirs(TFRECORDS_DIR)


def _image(path):

    with open(path, "rb") as f:
        return f.read()


def _dataset_exists(dataset_dir):
    for split_name in TFRECORDS_NAME_MAP.values():
        output_filename = os.path.join(dataset_dir, "{}_{}.tfrecords".format(TARGET_MODEL, split_name))
        if not tf.gfile.Exists(output_filename):
            return False
    return True


def _get_all_files(dataset_dir):
    file_list = []
    for filename in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, filename)
        file_list.append(path)
    return file_list


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def image_to_tfrecords(image_data, label):
    return tf.train.Example(features=tf.train.Features(feature={
        'image': bytes_feature(image_data),
        'label': bytes_feature(label),
    }))


def _convert_dataset(file_list, mode):

    output_filename = os.path.join(TFRECORDS_DIR, "{}_{}.tfrecords".format(TARGET_MODEL, TFRECORDS_NAME_MAP[mode]))
    with tf.python_io.TFRecordWriter(output_filename) as writer:
        for i, file_name in enumerate(file_list):
            try:
                sys.stdout.write('\r>> Converting image %d/%d ' % (i + 1, len(file_list)))
                sys.stdout.flush()
                image_data = _image(file_name)
                labels = re.search(REGEX_MAP[mode], file_name.split(PATH_SPLIT)[-1]).group()
                labels = labels.encode('utf-8')

                example = image_to_tfrecords(image_data, labels)
                writer.write(example.SerializeToString())

            except IOError as e:
                print('could not read:', file_list[1])
                print('error:', e)
                print('skip it \n')
    sys.stdout.write('\n')
    sys.stdout.flush()


if __name__ == '__main__':

    if _dataset_exists(TFRECORDS_DIR):
        print('Exists!')
    else:
        if isinstance(TRAINS_PATH, list):
            origin_dataset = []
            for trains_path in TRAINS_PATH:
                origin_dataset += [os.path.join(trains_path, trains) for trains in os.listdir(trains_path)]
        else:
            origin_dataset = [os.path.join(TRAINS_PATH, trains) for trains in os.listdir(TRAINS_PATH)]
        if HAS_TEST_SET:
            trains_dataset = origin_dataset
            if isinstance(TEST_PATH, list):
                test_dataset = []
                for test_path in TEST_PATH:
                    test_dataset += [os.path.join(test_path, test) for test in os.listdir(test_path)]
            else:
                test_dataset = [os.path.join(TEST_PATH, test) for test in os.listdir(TEST_PATH)]
        else:
            random.seed(_RANDOM_SEED)
            random.shuffle(origin_dataset)
            test_dataset = origin_dataset[:TEST_SET_NUM]
            trains_dataset = origin_dataset[TEST_SET_NUM:]

        _convert_dataset(test_dataset, mode=RunMode.Test)
        _convert_dataset(trains_dataset, mode=RunMode.Trains)
        print("Done!")
