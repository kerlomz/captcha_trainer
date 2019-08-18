#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>

import tensorflow as tf
from config import NUM_HIDDEN


class GRU(object):

    def __init__(self, inputs: tf.Tensor):
        self.inputs = inputs
        self.layer = None

    def build(self):
        with tf.compat.v1.variable_scope('GRU'):
            mask = tf.keras.layers.Masking()(self.inputs)
            self.layer = tf.keras.layers.GRU(
                units=NUM_HIDDEN * 2,
                return_sequences=True,
                input_shape=mask.shape
            )
            outputs = self.layer(mask)
        return outputs


class GRUcuDNN(object):

    def __init__(self, inputs: tf.Tensor):
        self.inputs = inputs
        self.layer = None

    def build(self):
        with tf.variable_scope('GRU'):
            mask = tf.keras.layers.Masking()(self.inputs)
            self.layer = tf.keras.layers.GRU(
                units=NUM_HIDDEN * 2,
                return_sequences=True,
                input_shape=mask.shape
            )
            outputs = self.layer(mask)
        return outputs
