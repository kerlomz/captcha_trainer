#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import tensorflow as tf
from network.utils import NetworkUtils
from config import IMAGE_CHANNEL


class CNN5(object):

    def __init__(self, inputs: tf.Tensor, utils: NetworkUtils):
        self.inputs = inputs
        self.utils = utils
        # (in_channels, out_channels)
        self.filters = [(IMAGE_CHANNEL, 32), (32, 64), (64, 128), (128, 128), (128, 64)]
        # (conv2d_strides, max_pool_strides)
        self.strides = [(1, 1), (1, 2), (1, 2), (1, 2), (1, 2)]
        self.filter_size = [7, 5, 3, 3, 3]

    def build(self):
        with tf.variable_scope('cnn'):
            x = self.inputs
            x = self.utils.cnn_layers(
                inputs=x,
                filter_size=self.filter_size,
                filters=self.filters,
                strides=self.strides
            )

            shape_list = x.get_shape().as_list()
            x = tf.reshape(x, [tf.shape(x)[0], -1, shape_list[2] * shape_list[3]])
            return x
