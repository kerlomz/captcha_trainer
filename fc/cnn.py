#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import tensorflow as tf
from tensorflow.python.keras.regularizers import l1_l2
from config import ModelConfig, RunMode
from exception import exception

from network.utils import NetworkUtils


class FullConnectedCNN(object):
    """
    CNN的输出层
    """
    def __init__(self, model_conf: ModelConfig, outputs):
        self.model_conf = model_conf

        self.max_label_num = self.model_conf.max_label_num
        if self.max_label_num == -1:
            exception(text="The scene must set the maximum number of label (MaxLabelNum)", code=-998)
        self.category_num = self.model_conf.category_num

        flatten = tf.keras.layers.Flatten()(outputs)
        shape_list = flatten.get_shape().as_list()

        # print(shape_list[1], self.max_label_num)
        outputs = tf.keras.layers.Reshape([self.max_label_num, int(shape_list[1] / self.max_label_num)])(flatten)
        self.outputs = tf.keras.layers.Dense(
            input_shape=outputs.shape,
            units=self.category_num,
        )(inputs=outputs)

        print("output to reshape ----------- ", self.outputs.shape)
        self.outputs = tf.keras.layers.Reshape([self.max_label_num, self.category_num])(self.outputs)

    def build(self):
        return self.outputs
