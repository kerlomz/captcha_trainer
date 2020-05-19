#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
# This network was temporarily suspended
import tensorflow as tf
from network.utils import NetworkUtils
from config import ModelConfig


class MobileNetV2(object):

    def __init__(self, model_conf: ModelConfig, inputs: tf.Tensor, utils: NetworkUtils):
        self.model_conf = model_conf
        self.inputs = inputs
        self.utils = utils
        self.loss_func = self.model_conf.loss_func
        self.last_block_filters = 1280
        self.padding = "SAME"

    def first_layer(self, inputs):
        x = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            kernel_initializer='he_normal',
            name='conv1')(inputs)
        x = tf.layers.batch_normalization(
            x,
            reuse=False,
            momentum=0.9,
            training=self.utils.is_training
        )
        # x = self.utils.BatchNormalization(name='bn_conv1', momentum=0.999)(x, training=self.utils.is_training)
        x = tf.keras.layers.LeakyReLU(0.01)(x)

        return x

    def pwise_block(self, inputs):
        x = tf.keras.layers.Conv2D(
            self.last_block_filters,
            kernel_size=1,
            use_bias=False,
            name='Conv_1')(inputs)
        x = tf.layers.batch_normalization(
            x,
            reuse=False,
            momentum=0.9,
            training=self.utils.is_training
        )

        x = tf.keras.layers.ReLU(6., name='out_relu')(x)
        return x

    def build(self):

        with tf.keras.backend.name_scope('MobileNetV2'):

            x = self.first_layer(self.inputs)

            x = self.utils.inverted_res_block(x, filters=16, stride=1, expansion=1, block_id=0)

            x = self.utils.inverted_res_block(x, filters=24, stride=2, expansion=6, block_id=1)
            x = self.utils.inverted_res_block(x, filters=24, stride=1, expansion=6, block_id=2)

            x = self.utils.inverted_res_block(x, filters=32, stride=2, expansion=6, block_id=3)
            x = self.utils.inverted_res_block(x, filters=32, stride=1, expansion=6, block_id=4)
            x = self.utils.inverted_res_block(x, filters=32, stride=1, expansion=6, block_id=5)

            x = self.utils.inverted_res_block(x, filters=64, stride=2, expansion=6, block_id=6)
            x = self.utils.inverted_res_block(x, filters=64, stride=1, expansion=6, block_id=7)
            x = self.utils.inverted_res_block(x, filters=64, stride=1, expansion=6, block_id=8)
            x = self.utils.inverted_res_block(x, filters=64, stride=1, expansion=6, block_id=9)

            x = self.utils.inverted_res_block(x, filters=96, stride=1, expansion=6, block_id=10)
            x = self.utils.inverted_res_block(x, filters=96, stride=1, expansion=6, block_id=11)
            x = self.utils.inverted_res_block(x, filters=96, stride=1, expansion=6, block_id=12)

            x = self.utils.inverted_res_block(x, filters=160, stride=2, expansion=6, block_id=13)
            x = self.utils.inverted_res_block(x, filters=160, stride=1, expansion=6, block_id=14)
            x = self.utils.inverted_res_block(x, filters=160, stride=1, expansion=6, block_id=15)

            x = self.utils.inverted_res_block(x, filters=320, stride=1, expansion=6, block_id=16)

            x = self.pwise_block(x)

            shape_list = x.get_shape().as_list()
            print("x.get_shape()", shape_list)

            return self.utils.reshape_layer(x, self.loss_func, shape_list)
