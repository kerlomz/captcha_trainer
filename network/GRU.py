#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>

import tensorflow as tf
from tensorflow.python.keras.regularizers import l2
from config import NUM_HIDDEN, RunMode
from network.utils import NetworkUtils


class GRU(object):

    def __init__(self, inputs: tf.Tensor, utils: NetworkUtils):
        self.inputs = inputs
        self.utils = utils
        self.layer = None

    def build(self):
        with tf.compat.v1.variable_scope('GRU'):
            mask = tf.keras.layers.Masking()(self.inputs)
            self.layer = tf.keras.layers.GRU(
                units=NUM_HIDDEN * 2,
                return_sequences=True,
                input_shape=mask.shape,
                reset_after=True,
                # implementation=2,
                recurrent_regularizer=l2(0.01),
                kernel_regularizer=l2(0.01),
                bias_regularizer=l2(0.005),
                trainable=self.utils.training,
            )
            outputs = self.layer(mask, training=self.utils.training)
        return outputs


class BiGRU(object):

    def __init__(self, inputs: tf.Tensor, utils: NetworkUtils):
        self.inputs = inputs
        self.utils = utils

    def build(self):
        with tf.compat.v1.variable_scope('BiGRU'):
            mask = tf.keras.layers.Masking()(self.inputs)
            forward_layer = tf.keras.layers.GRU(
                units=NUM_HIDDEN * 2,
                return_sequences=True,
                input_shape=mask.shape,
                reset_after=True,
                kernel_regularizer=l2(0.01),
                bias_regularizer=l2(0.005),
                trainable=self.utils.training,
            )
            backward_layer = tf.keras.layers.GRU(
                units=NUM_HIDDEN * 2,
                return_sequences=True,
                input_shape=mask.shape,
                reset_after=True,
                go_backwards=True,
                kernel_regularizer=l2(0.01),
                bias_regularizer=l2(0.005),
                trainable=self.utils.training,
            )
            forward = forward_layer(mask, training=self.utils.training)
            backward = backward_layer(mask, training=self.utils.training)
            outputs = tf.keras.layers.Concatenate(axis=2)([forward, backward])
        return outputs


class GRUcuDNN(object):

    def __init__(self, inputs: tf.Tensor, utils: NetworkUtils):
        self.inputs = inputs
        self.utils = utils
        self.training = self.utils.mode == RunMode.Trains
        self.layer = None

    def build(self):
        with tf.variable_scope('GRU'):
            mask = tf.keras.layers.Masking()(self.inputs)
            self.layer = tf.keras.layers.GRU(
                units=NUM_HIDDEN * 2,
                return_sequences=True,
                input_shape=mask.shape,
                reset_after=True
            )
            outputs = self.layer(mask, training=self.training)
        return outputs
