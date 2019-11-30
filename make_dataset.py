#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import random
from tqdm import tqdm
import tensorflow as tf
from config import *
from constants import RunMode

_RANDOM_SEED = 0

TFRECORDS_TYPE = [
    RunMode.Trains,
    RunMode.Validation
]


class DataSets:

    """此类用于打包数据集为TFRecords格式"""
    def __init__(self, model: ModelConfig, ):
        self.model = model
        if not os.path.exists(self.model.dataset_root_path):
            os.makedirs(self.model.dataset_root_path)

    @staticmethod
    def read_image(path):
        """
        读取图片
        :param path: 图片路径
        :return:
        """
        with open(path, "rb") as f:
            return f.read()

    def dataset_exists(self):
        """数据集是否存在判断函数"""
        for split_name in TFRECORDS_TYPE:
            output_filename = os.path.join(self.model.dataset_root_path, "{}_{}.tfrecords".format(self.model.model_name, split_name.value))
            if not tf.io.gfile.exists(output_filename):
                return False
        return True

    @staticmethod
    def bytes_feature(values):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

    def input_to_tfrecords(self, input_data, label):
        return tf.train.Example(features=tf.train.Features(feature={
            'input': self.bytes_feature(input_data),
            'label': self.bytes_feature(label),
        }))

    def convert_dataset(self, file_list, mode):
        output_filename = os.path.join(self.model.dataset_root_path, "{}_{}.tfrecords".format(self.model.model_name, mode.value))
        with tf.io.TFRecordWriter(output_filename) as writer:
            pbar = tqdm(file_list)
            for i, file_name in enumerate(pbar):
                try:
                    image_data = self.read_image(file_name)
                    labels = re.search(self.model.extract_regex, file_name.split(PATH_SPLIT)[-1])
                    if labels:
                        labels = labels.group()
                    else:
                        raise NameError('invalid filename {}'.format(file_name))

                    # labels = "".join([CH_NUMBER_MAP[c] if c in CH_NUMBER else c for c in labels])
                    labels = labels.encode('utf-8')

                    example = self.input_to_tfrecords(image_data, labels)
                    writer.write(example.SerializeToString())
                    pbar.set_description('[Processing dataset %s] [filename: %s]' % (mode, file_name))

                except IOError as e:
                    print('could not read:', file_list[1])
                    print('error:', e)
                    print('skip it \n')

    def make_dataset(self):
        if self.dataset_exists():
            print('Exists!')
        else:

            if isinstance(self.model.dataset_path, list):
                origin_dataset = []
                for trains_path in self.model.dataset_path:
                    origin_dataset += [os.path.join(trains_path, trains) for trains in os.listdir(trains_path)]
            else:
                origin_dataset = [os.path.join(self.model.dataset_path, trains) for trains in os.listdir(self.model.dataset_path)]

            random.seed(_RANDOM_SEED)
            random.shuffle(origin_dataset)
            validation_dataset = origin_dataset[:self.model.validation_set_num]
            trains_dataset = origin_dataset[self.model.validation_set_num:]

            self.convert_dataset(validation_dataset, mode=RunMode.Validation)
            self.convert_dataset(trains_dataset, mode=RunMode.Trains)
            print("Done!")


if __name__ == '__main__':
    m = ModelConfig("a")
    DataSets(m).make_dataset()
