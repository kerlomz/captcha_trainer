#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
# This network was temporarily suspended
import tensorflow as tf
from network.utils import NetworkUtils
from config import ModelConfig


class DenseNet(object):

    def __init__(self, model_conf: ModelConfig, inputs: tf.Tensor, utils: NetworkUtils):
        self.model_conf = model_conf
        self.inputs = inputs
        self.utils = utils
        self.loss_func = self.model_conf.loss_func
        self.type = {
            '121': [6, 12, 24, 16],
            '169': [6, 12, 32, 32],
            '201': [6, 12, 48, 32]
        }
        self.blocks = self.type['121']
        self.padding = "SAME"

    def build(self):

        with tf.keras.backend.name_scope('DenseNet'):

            x = tf.keras.layers.Conv2D(64, 3, strides=2, use_bias=False, name='conv1/conv', padding='same')(self.inputs)
            x = tf.layers.batch_normalization(
                x,
                epsilon=1.001e-5,
                axis=3,
                reuse=False,
                momentum=0.9,
                name='conv1/bn',
                training=self.utils.is_training,
            )

            x = tf.keras.layers.LeakyReLU(0.01, name='conv1/relu')(x)
            x = tf.keras.layers.MaxPooling2D(3, strides=2, name='pool1', padding='same')(x)
            x = self.utils.dense_block(x, self.blocks[0], name='conv2')
            x = self.utils.transition_block(x, 0.5, name='pool2')
            x = self.utils.dense_block(x, self.blocks[1], name='conv3')
            x = self.utils.transition_block(x, 0.5, name='pool3')
            x = self.utils.dense_block(x, self.blocks[2], name='conv4')
            x = self.utils.transition_block(x, 0.5, name='pool4')
            x = self.utils.dense_block(x, self.blocks[3], name='conv5')
            x = tf.layers.batch_normalization(
                x,
                epsilon=1.001e-5,
                axis=3,
                reuse=False,
                momentum=0.9,
                name='bn',
                training=self.utils.is_training,
            )

            x = tf.keras.layers.LeakyReLU(0.01, name='conv6/relu')(x)

            shape_list = x.get_shape().as_list()
            print("x.get_shape()", shape_list)

            return self.utils.reshape_layer(x, self.loss_func, shape_list)
