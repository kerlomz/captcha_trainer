#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>

import tensorflow as tf
from config import RunMode, ModelConfig
from network.utils import NetworkUtils


class GRU(object):

    def __init__(self, model_conf: ModelConfig, inputs: tf.Tensor, utils: NetworkUtils):
        """
        :param model_conf: 配置
        :param inputs: 网络上一层输入tf.keras.layers.Input/tf.Tensor类型
        :param utils: 网络工具类
        """
        self.model_conf = model_conf
        self.inputs = inputs
        self.utils = utils
        self.layer = None

    def build(self):
        """
        循环层构建参数
        :return: 返回循环层的输出层
        """
        with tf.keras.backend.name_scope('GRU'):
            mask = tf.keras.layers.Masking()(self.inputs)
            self.layer = tf.keras.layers.GRU(
                units=self.model_conf.units_num * 2,
                return_sequences=True,
                input_shape=mask.shape,
                # reset_after=True,
            )
            outputs = self.layer(mask, training=self.utils.is_training)
        return outputs


class BiGRU(object):

    def __init__(self, model_conf: ModelConfig, inputs: tf.Tensor, utils: NetworkUtils):
        self.model_conf = model_conf
        self.inputs = inputs
        self.utils = utils
        self.training = self.utils.mode == RunMode.Trains
        self.layer = None

    def build(self):
        with tf.keras.backend.name_scope('BiGRU'):
            mask = tf.keras.layers.Masking()(self.inputs)
            self.layer = tf.keras.layers.Bidirectional(
                layer=tf.keras.layers.GRU(
                    units=self.model_conf.units_num,
                    return_sequences=True,
                ),
                input_shape=mask.shape,
                trainable=self.utils.is_training
            )
            outputs = self.layer(mask, training=self.training)
        return outputs


class GRUcuDNN(object):

    def __init__(self, model_conf: ModelConfig, inputs: tf.Tensor, utils: NetworkUtils):
        self.model_conf = model_conf
        self.inputs = inputs
        self.utils = utils
        self.training = self.utils.mode == RunMode.Trains
        self.layer = None

    def build(self):
        with tf.keras.backend.name_scope('GRU'):
            mask = tf.keras.layers.Masking()(self.inputs)
            self.layer = tf.keras.layers.GRU(
                units=self.model_conf.units_num * 2,
                return_sequences=True,
                input_shape=mask.shape,
                reset_after=True
            )
            outputs = self.layer(mask, training=self.training)
        return outputs
