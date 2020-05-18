#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import tensorflow as tf
from config import RunMode, ModelConfig
from network.utils import NetworkUtils


class LSTM(object):
    """
    LSTM 网络实现
    """
    def __init__(self, model_conf: ModelConfig, inputs: tf.Tensor, utils: NetworkUtils):
        """
        :param model_conf: 配置
        :param inputs: 网络上一层输入 tf.keras.layers.Input / tf.Tensor 类型
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
        with tf.keras.backend.name_scope('LSTM'):
            mask = tf.keras.layers.Masking()(self.inputs)
            self.layer = tf.keras.layers.LSTM(
                units=self.model_conf.units_num * 2,
                return_sequences=True,
                input_shape=mask.shape,
                dropout=0.2,
                recurrent_dropout=0.1
            )
            outputs = self.layer(mask, training=self.utils.is_training)
        return outputs


class BiLSTM(object):

    def __init__(self, model_conf: ModelConfig, inputs: tf.Tensor, utils: NetworkUtils):
        """同上"""
        self.model_conf = model_conf
        self.inputs = inputs
        self.utils = utils
        self.training = self.utils.mode == RunMode.Trains
        self.layer = None

    def build(self):
        """同上"""
        with tf.keras.backend.name_scope('BiLSTM'):
            mask = tf.keras.layers.Masking()(self.inputs)
            self.layer = tf.keras.layers.Bidirectional(
                layer=tf.keras.layers.LSTM(
                    units=self.model_conf.units_num,
                    return_sequences=True,
                ),
                input_shape=mask.shape,
            )
            outputs = self.layer(mask, training=self.utils.is_training)
        return outputs


class LSTMcuDNN(object):

    def __init__(self, model_conf: ModelConfig, inputs: tf.Tensor, utils: NetworkUtils):
        """同上"""
        self.model_conf = model_conf
        self.inputs = inputs
        self.utils = utils
        self.training = self.utils.mode == RunMode.Trains
        self.layer = None

    def build(self):
        """同上"""
        with tf.keras.backend.name_scope('LSTM'):
            self.layer = tf.keras.layers.CuDNNLSTM(
                units=self.model_conf.units_num * 2,
                return_sequences=True,
            )
            outputs = self.layer(self.inputs, training=self.training)
        return outputs


class BiLSTMcuDNN(object):

    def __init__(self, model_conf: ModelConfig, inputs: tf.Tensor, utils: NetworkUtils):
        """同上"""
        self.model_conf = model_conf
        self.inputs = inputs
        self.utils = utils
        self.training = self.utils.mode == RunMode.Trains
        self.layer = None

    def build(self):
        """同上"""
        with tf.keras.backend.name_scope('BiLSTM'):
            self.layer = tf.keras.layers.Bidirectional(
                layer=tf.keras.layers.CuDNNLSTM(
                    units=self.model_conf.units_num,
                    return_sequences=True
                )
            )
            outputs = self.layer(self.inputs, training=self.training)
        return outputs
