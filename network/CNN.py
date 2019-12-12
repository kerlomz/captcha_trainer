#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import tensorflow as tf
from network.utils import NetworkUtils
from config import *
from tensorflow.python.keras.regularizers import l1, l2, l1_l2


class CNN5(object):

    """
    CNN5网络的实现
    """
    def __init__(self, model_conf: ModelConfig, inputs: tf.Tensor, utils: NetworkUtils):
        """
        :param model_conf: 从配置文件
        :param inputs: 网络上一层输入tf.keras.layers.Input/tf.Tensor类型
        :param utils: 网络工具类
        """
        self.model_conf = model_conf
        self.inputs = inputs
        self.utils = utils
        self.loss_func = self.model_conf.loss_func

    def build(self):
        with tf.compat.v1.variable_scope("CNN5"):
            x = self.utils.cnn_layer(0, inputs=self.inputs, kernel_size=7, filters=32, strides=(1, 1))
            x = self.utils.cnn_layer(1, inputs=x, kernel_size=5, filters=64, strides=(1, 2))
            x = self.utils.cnn_layer(2, inputs=x, kernel_size=3, filters=128, strides=(1, 2))
            x = self.utils.cnn_layer(3, inputs=x, kernel_size=3, filters=129, strides=(1, 2))
            x = self.utils.cnn_layer(4, inputs=x, kernel_size=3, filters=64, strides=(1, 2))
            shape_list = x.get_shape().as_list()
            print("x.get_shape()", shape_list)
            return self.utils.reshape_layer(x, self.loss_func, shape_list)


class CNNX(object):
    """暂时不用管，设计到一半的一个网络结构"""
    def __init__(self, model_conf: ModelConfig, inputs: tf.Tensor, utils: NetworkUtils):
        self.model_conf = model_conf
        self.inputs = inputs
        self.utils = utils
        self.loss_func = self.model_conf.loss_func

    def block(self, inputs, filters, kernel_size, strides, dilation_rate=(1, 1)):
        inputs = tf.keras.layers.Conv2D(
            filters=filters,
            dilation_rate=dilation_rate,
            kernel_size=kernel_size,
            strides=strides,
            kernel_regularizer=l1(0.1),
            kernel_initializer=self.utils.msra_initializer(kernel_size, filters),
            padding='SAME',
        )(inputs)
        inputs = tf.layers.BatchNormalization(
            fused=True,
            epsilon=1.001e-5,
        )(inputs, training=self.utils.training)
        inputs = tf.keras.layers.LeakyReLU(0.01)(inputs)
        return inputs

    def depth_block(self, input_tensor, kernel_size=1, depth_multiplier=2, strides=1):
        x = tf.keras.layers.DepthwiseConv2D(
            depthwise_regularizer=l2(0.1),
            strides=strides,
            padding='SAME',
            kernel_size=kernel_size,
            depth_multiplier=depth_multiplier
        )(input_tensor)
        x = tf.layers.BatchNormalization(
            fused=True,
            epsilon=1e-3,
            momentum=0.999,
        )(x, training=self.utils.training)
        x = tf.keras.layers.LeakyReLU(0.01)(x)
        x = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=1,
            padding='SAME',
        )(x)
        x = tf.keras.layers.BatchNormalization(
            epsilon=1e-3,
            momentum=0.999,
        )(x)
        return x

    def build(self):
        with tf.compat.v1.variable_scope('CNNX'):
            x = self.inputs

            x = self.block(x, filters=32, kernel_size=7, strides=1)
            x = self.block(x, filters=64, kernel_size=5, strides=1)

            max_pool0 = tf.keras.layers.MaxPooling2D(
                pool_size=(1, 2),
                strides=2,
                padding='same')(x)
            max_pool1 = tf.keras.layers.MaxPooling2D(
                pool_size=(3, 2),
                strides=2,
                padding='same')(x)
            max_pool2 = tf.keras.layers.MaxPooling2D(
                pool_size=(5, 2),
                strides=2,
                padding='same')(x)
            max_pool3 = tf.keras.layers.MaxPooling2D(
                pool_size=(7, 2),
                strides=2,
                padding='same')(x)

            multi_scale_pool = tf.keras.layers.Add()([max_pool0, max_pool1, max_pool2, max_pool3])
            x = self.utils.dense_block(multi_scale_pool, 1, name='conv2')
            x = self.utils.transition_block(x, 0.5, name='pool2')
            x1 = self.depth_block(x, kernel_size=3, strides=2, depth_multiplier=2)
            x2 = tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2),
                strides=2,
                padding='same')(x)
            x = tf.keras.layers.Concatenate()([x2, x1])
            x = self.block(x, filters=64, kernel_size=3, strides=1)
            x = tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2),
                strides=1,
                padding='same')(x)
            shape_list = x.get_shape().as_list()
            print("x.get_shape()", shape_list)
            return self.utils.reshape_layer(x, self.loss_func, shape_list)
