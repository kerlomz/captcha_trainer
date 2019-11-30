#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
# This network was temporarily suspended
import tensorflow as tf
from network.utils import NetworkUtils
from config import ModelConfig
from constants import LossFunction


class DenseNet(object):

    def __init__(self, model_conf: ModelConfig, inputs: tf.Tensor, utils: NetworkUtils):
        self.model_conf = model_conf
        self.inputs = inputs
        self.utils = utils
        self.loss_func = self.model_conf.loss_func
        self.type = {
            '121': [6, 12, 24, 16],
            '169': [6, 12, 32, 32],
            '201': [6, 12, 48, 32]
        }
        self.blocks = self.type['121']
        self.padding = "SAME"

    def build(self):

        with tf.variable_scope('DenseNet'):

            # # Keras Implementation Version
            # x = tf.keras.applications.densenet.DenseNet121(
            #     include_top=False,
            #     weights=None,
            #     pooling=None,
            #     input_tensor=tf.keras.Input(
            #         tensor=self.inputs,
            #         shape=self.inputs.get_shape().as_list()
            #     )
            # )(self.inputs)

            # TensorFlow Implementation Version
            x = tf.keras.layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(self.inputs)
            x = tf.layers.BatchNormalization(axis=3, epsilon=1.001e-5, name='conv1/bn')(x, training=self.utils.training)
            x = tf.keras.layers.LeakyReLU(0.01, name='conv1/relu')(x)
            x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
            x = tf.keras.layers.MaxPooling2D(3, strides=2, name='pool1')(x)
            x, self.utils.dense_block(x, self.blocks[0], name='conv2')
            x, self.utils.transition_block(x, 0.5, name='pool2')
            x, self.utils.dense_block(x, self.blocks[1], name='conv3')
            x, self.utils.transition_block(x, 0.5, name='pool3')
            x, self.utils.dense_block(x, self.blocks[2], name='conv4')
            x, self.utils.transition_block(x, 0.5, name='pool4')
            x, self.utils.dense_block(x, self.blocks[3], name='conv5')
            x = tf.layers.BatchNormalization(axis=3, epsilon=1.001e-5, name='bn')(x, training=self.utils.training)

            shape_list = x.get_shape().as_list()
            if self.loss_func == LossFunction.CTC:
                x = tf.keras.layers.Reshape(target_shape=[-1, shape_list[2] * shape_list[3]])(x)
            elif self.loss_func == LossFunction.CrossEntropy:
                x = tf.keras.layers.Reshape(target_shape=[shape_list[1], shape_list[2] * shape_list[3]])(x)
            return x
