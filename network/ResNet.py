#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>

import tensorflow as tf
from tensorflow.contrib.slim import nets
from network.utils import NetworkUtils, RunMode
from constants import LossFunction
from config import ModelConfig

slim = tf.contrib.slim


class ResNetUtils(object):

    def __init__(self, utils: NetworkUtils):
        self.utils = utils

    def first_layer(self, inputs):
        # x = tf.keras.layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(inputs)
        x = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding='same',
            kernel_initializer='he_normal',
            name='conv1')(inputs)

        x = tf.layers.BatchNormalization(name='bn_conv1')(x, training=self.utils.training)
        x = tf.keras.layers.LeakyReLU(0.01)(x)
        # x = tf.keras.layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
        x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same',)(x)
        return x


class ResNet50(object):

    def __init__(self, model_conf: ModelConfig, inputs: tf.Tensor, utils: NetworkUtils):
        self.model_conf = model_conf
        self.inputs = inputs
        self.utils = utils
        self.loss_func = self.model_conf.loss_func

    def build(self):

        x = ResNetUtils(self.utils).first_layer(self.inputs)
        x = self.utils.residual_building_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = self.utils.identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = self.utils.identity_block(x, 3, [64, 64, 256], stage=2, block='c')

        x = self.utils.residual_building_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = self.utils.identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = self.utils.identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = self.utils.identity_block(x, 3, [128, 128, 512], stage=3, block='d')

        x = self.utils.residual_building_block(x, 3, [256, 256, 1024], stage=4, block='a')
        x = self.utils.identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        x = self.utils.identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        x = self.utils.identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        x = self.utils.identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        x = self.utils.identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

        x = self.utils.residual_building_block(x, 3, [512, 512, 2048], stage=5, block='a', strides=(1, 1))
        x = self.utils.identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = self.utils.identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

        print("x.get_shape()", x.get_shape())
        shape_list = x.get_shape().as_list()
        return self.utils.reshape_layer(x, self.loss_func, shape_list)


class ResNetTiny(object):

    def __init__(self, model_conf: ModelConfig, inputs: tf.Tensor, utils: NetworkUtils):
        self.model_conf = model_conf
        self.inputs = inputs
        self.utils = utils
        self.loss_func = self.model_conf.loss_func

    def build(self):

        x = ResNetUtils(self.utils).first_layer(self.inputs)
        x = self.utils.residual_building_block(x, 3, [64, 64, 128], stage=2, block='a', strides=(1, 1))
        x = self.utils.identity_block(x, 3, [64, 64, 128], stage=2, block='b')

        x = self.utils.residual_building_block(x, 3, [128, 128, 256], stage=3, block='a')
        x = self.utils.identity_block(x, 3, [128, 128, 256], stage=3, block='b')

        x = self.utils.residual_building_block(x, 3, [256, 256, 512], stage=4, block='a')
        x = self.utils.identity_block(x, 3, [256, 256, 512], stage=4, block='b')

        x = self.utils.residual_building_block(x, 3, [512, 512, 1024], stage=5, block='a', strides=(1, 1))
        x = self.utils.identity_block(x, 3, [512, 512, 1024], stage=5, block='b')

        print("x.get_shape()", x.get_shape())
        shape_list = x.get_shape().as_list()
        return self.utils.reshape_layer(x, self.loss_func, shape_list)
