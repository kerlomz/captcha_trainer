#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import tensorflow as tf
from tensorflow.python.keras.regularizers import l1_l2
from config import RunMode, ModelConfig
from network.utils import NetworkUtils


class FullConnectedRNN(object):

    def __init__(self, model_conf: ModelConfig, mode: RunMode, outputs):
        self.model_conf = model_conf
        self.utils = NetworkUtils(mode)

        self.dense = lambda: tf.keras.layers.Dense(
            units=self.model_conf.category_num + 2,
            kernel_initializer=tf.keras.initializers.glorot_normal(seed=None),
            kernel_regularizer=l1_l2(l1=0.001, l2=0.01),
            bias_initializer='zeros',
        )
        self.time_distributed = lambda: tf.keras.layers.TimeDistributed(
            layer=self.dense(),
            name='predict',
        )(inputs=outputs, training=self.utils.training)

        self.outputs = self.time_distributed()
        self.predict = tf.keras.backend.permute_dimensions(self.outputs, pattern=(1, 0, 2))

    def build(self):
        return self.predict
