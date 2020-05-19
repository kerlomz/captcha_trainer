#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import tensorflow as tf
from network.utils import NetworkUtils
from config import ModelConfig
from tensorflow.python.keras.regularizers import l1


class CNN5(object):

    """
    CNN5网络的实现
    """
    def __init__(self, model_conf: ModelConfig, inputs: tf.Tensor, utils: NetworkUtils):
        """
        :param model_conf: 从配置文件
        :param inputs: 网络上一层输入 tf.keras.layers.Input / tf.Tensor 类型
        :param utils: 网络工具类
        """
        self.model_conf = model_conf
        self.inputs = inputs
        self.utils = utils
        self.loss_func = self.model_conf.loss_func

    def build(self):
        with tf.keras.backend.name_scope("CNN5"):
            x = self.utils.cnn_layer(0, inputs=self.inputs, kernel_size=7, filters=32, strides=(1, 1))
            x = self.utils.cnn_layer(1, inputs=x, kernel_size=5, filters=64, strides=(1, 2))
            x = self.utils.cnn_layer(2, inputs=x, kernel_size=3, filters=128, strides=(1, 2))
            x = self.utils.cnn_layer(3, inputs=x, kernel_size=3, filters=128, strides=(1, 2))
            x = self.utils.cnn_layer(4, inputs=x, kernel_size=3, filters=64, strides=(1, 2))
            shape_list = x.get_shape().as_list()
            print("x.get_shape()", shape_list)
            return self.utils.reshape_layer(x, self.loss_func, shape_list)


class CNNX(object):

    """ 网络结构 """
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
        inputs = tf.layers.batch_normalization(
            inputs,
            reuse=False,
            momentum=0.9,
            training=self.utils.is_training
        )
        inputs = self.utils.hard_swish(inputs)
        return inputs

    def build(self):
        with tf.keras.backend.name_scope('CNNX'):
            x = self.inputs

            x = self.block(x, filters=16, kernel_size=7, strides=1)

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

            x = self.block(multi_scale_pool, filters=32, kernel_size=5, strides=1)

            x1 = self.utils.inverted_res_block(x, filters=16, stride=2, expansion=6, block_id=1)
            x1 = self.utils.inverted_res_block(x1, filters=16, stride=1, expansion=6, block_id=2)

            x2 = tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2),
                strides=2,
                padding='same')(x)
            x = tf.keras.layers.Concatenate()([x2, x1])

            x = self.utils.inverted_res_block(x, filters=32, stride=2, expansion=6, block_id=3)
            x = self.utils.inverted_res_block(x, filters=32, stride=1, expansion=6, block_id=4)

            x = self.utils.dense_block(x, 2, name='dense_block')

            x = self.utils.inverted_res_block(x, filters=64, stride=1, expansion=6, block_id=5)

            shape_list = x.get_shape().as_list()
            print("x.get_shape()", shape_list)

            return self.utils.reshape_layer(x, self.loss_func, shape_list)
