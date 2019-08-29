#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import tensorflow as tf
from config import NUM_HIDDEN, RunMode
from network.utils import NetworkUtils


class LSTM(object):

    def __init__(self, inputs: tf.Tensor, utils: NetworkUtils):
        self.inputs = inputs
        self.utils = utils
        self.training = self.utils.mode == RunMode.Trains
        self.layer = None

    def build(self):
        with tf.compat.v1.variable_scope('LSTM'):
            mask = tf.keras.layers.Masking()(self.inputs)
            self.layer = tf.keras.layers.LSTM(
                units=NUM_HIDDEN * 2,
                return_sequences=True,
                input_shape=mask.shape,
                dropout=0.2,
                recurrent_dropout=0.1
            )
            outputs = self.layer(mask, training=self.training)
        return outputs


class BiLSTM(object):

    def __init__(self, inputs: tf.Tensor, utils: NetworkUtils):
        self.inputs = inputs
        self.utils = utils
        self.training = self.utils.mode == RunMode.Trains
        self.layer = None

    def build(self):
        with tf.variable_scope('BiLSTM'):
            mask = tf.keras.layers.Masking()(self.inputs)
            self.layer = tf.keras.layers.Bidirectional(
                layer=tf.keras.layers.LSTM(
                    units=NUM_HIDDEN,
                    return_sequences=True,
                ),
                input_shape=mask.shape
            )
            outputs = self.layer(mask, training=self.training)
        return outputs


class LSTMcuDNN(object):

    def __init__(self, inputs: tf.Tensor, utils: NetworkUtils):
        self.inputs = inputs
        self.utils = utils
        self.training = self.utils.mode == RunMode.Trains
        self.layer = None

    def build(self):
        with tf.variable_scope('LSTM'):
            self.layer = tf.keras.layers.CuDNNLSTM(
                units=NUM_HIDDEN * 2,
                return_sequences=True,
            )
            outputs = self.layer(self.inputs, training=self.training)
        return outputs


class BiLSTMcuDNN(object):

    def __init__(self, inputs: tf.Tensor, utils: NetworkUtils):
        self.inputs = inputs
        self.utils = utils
        self.training = self.utils.mode == RunMode.Trains
        self.layer = None

    def build(self):
        with tf.variable_scope('BiLSTM'):
            self.layer = tf.keras.layers.Bidirectional(
                layer=tf.keras.layers.CuDNNLSTM(
                    units=NUM_HIDDEN,
                    return_sequences=True
                )
            )
            outputs = self.layer(self.inputs, training=self.training)
        return outputs
