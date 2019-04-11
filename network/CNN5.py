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
        self.filters = [32, 64, 128, 128, 64]
        self.strides = [1, 2]

    def build(self):
        with tf.variable_scope('cnn'):
            with tf.variable_scope('unit-1'):
                x = self.utils.conv2d(self.inputs, 'cnn-1', 7, IMAGE_CHANNEL, self.filters[0], self.strides[0])
                x = self.utils.batch_norm('bn1', x)
                x = self.utils.leaky_relu(x, 0.01)
                x = self.utils.max_pool(x, 2, self.strides[0])

            with tf.variable_scope('unit-2'):
                x = self.utils.conv2d(x, 'cnn-2', 5, self.filters[0], self.filters[1], self.strides[0])
                x = self.utils.batch_norm('bn2', x)
                x = self.utils.leaky_relu(x, 0.01)
                x = self.utils.max_pool(x, 2, self.strides[1])

            with tf.variable_scope('unit-3'):
                x = self.utils.conv2d(x, 'cnn-3', 3, self.filters[1], self.filters[2], self.strides[0])
                x = self.utils.batch_norm('bn3', x)
                x = self.utils.leaky_relu(x, 0.01)
                x = self.utils.max_pool(x, 2, self.strides[1])

            with tf.variable_scope('unit-4'):
                x = self.utils.conv2d(x, 'cnn-4', 3, self.filters[2], self.filters[3], self.strides[0])
                x = self.utils.batch_norm('bn4', x)
                x = self.utils.leaky_relu(x, 0.01)
                x = self.utils.max_pool(x, 2, self.strides[1])

            with tf.variable_scope('unit-5'):
                x = self.utils.conv2d(x, 'cnn-5', 3, self.filters[3], self.filters[4], self.strides[0])
                x = self.utils.batch_norm('bn5', x)
                x = self.utils.leaky_relu(x, 0.01)
                x = self.utils.max_pool(x, 2, self.strides[1])

            shape_list = x.get_shape().as_list()
            x = tf.reshape(x, [-1, shape_list[1], shape_list[2] * shape_list[3]])
            return x
