#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import tensorflow as tf
from network.utils import NetworkUtils


class DenseNet(object):

    def __init__(self, inputs: tf.Tensor, utils: NetworkUtils):
        self.inputs = inputs
        self.utils = utils
        self.nb_filter = 12
        self.strides = (2, 2)
        self.kernel_size = 5
        self.padding = "SAME"

    def build(self):
        with tf.variable_scope('DenseNet'):
            x = tf.layers.conv2d(
                inputs=self.inputs,
                filters=self.nb_filter,
                kernel_size=self.kernel_size,
                strides=self.strides,
                padding=self.padding,
                use_bias=False
            )
            x, nb_filter = self.utils.dense_block(x, 8, 8, self.nb_filter)
            x, nb_filter = self.utils.transition_block(x, 128, pool_type=2)
            x, nb_filter = self.utils.dense_block(x, 8, 8, nb_filter)
            x, nb_filter = self.utils.transition_block(x, 128, pool_type=3)
            x, nb_filter = self.utils.dense_block(x, 8, 8, nb_filter)

            shape_list = x.get_shape().as_list()
            x = tf.reshape(x, [-1, shape_list[1], shape_list[2] * shape_list[3]])
            return x
