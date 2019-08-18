#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
# This network was temporarily suspended
import tensorflow as tf
from network.utils import NetworkUtils, RunMode


class DenseNet(object):

    def __init__(self, inputs: tf.Tensor, utils: NetworkUtils):
        self.inputs = inputs
        self.utils = utils
        self.blocks = [6, 12, 24, 16]
        self.strides = (2, 2)
        self.kernel_size = 5
        self.padding = "SAME"

    def build(self):

        with tf.variable_scope('DenseNet'):

            # Keras Implementation Version
            # x = tf.keras.applications.densenet.DenseNet121(
            #     include_top=False,
            #     weights=None,
            #     pooling=None,
            #     input_tensor=tf.keras.Input(
            #         tensor=self.inputs,
            #         shape=self.inputs.get_shape().as_list()
            #     )
            # )(self.inputs)

            # TensorFlow Implementation Version
            x = tf.keras.layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(self.inputs)
            x = tf.keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5, name='conv1/bn')(x)
            x = tf.keras.layers.Activation('relu', name='conv1/relu')(x)
            x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
            x = tf.keras.layers.MaxPooling2D(3, strides=2, name='pool1')(x)
            x, self.utils.dense_block(x, self.blocks[0], name='conv2')
            x, self.utils.transition_block(x, 0.5, name='pool2')
            x, self.utils.dense_block(x, self.blocks[1], name='conv3')
            x, self.utils.transition_block(x, 0.5, name='pool3')
            x, self.utils.dense_block(x, self.blocks[2], name='conv4')
            x, self.utils.transition_block(x, 0.5, name='pool4')
            x, self.utils.dense_block(x, self.blocks[3], name='conv5')
            x = tf.keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5, name='bn')(x)

            shape_list = x.get_shape().as_list()
            x = tf.reshape(x, [tf.shape(x)[0], -1, shape_list[2] * shape_list[3]])
            return x
