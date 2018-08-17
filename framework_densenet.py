#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>

import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import batch_norm, flatten
import numpy as np
from config import *

x = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH], name='input')
label = tf.placeholder(tf.float32, [None, MAX_CAPTCHA_LEN * CHAR_SET_LEN], name='label')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
training_flag = tf.placeholder(tf.bool)
dropout_rate = 0
class_num = CHAR_SET_LEN * MAX_CAPTCHA_LEN


def conv_layer(_input, _filter, kernel, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=_input, filters=_filter, kernel_size=kernel, strides=stride, padding='SAME')
        return network


def global_average_pooling(_x, stride=1):
    width = np.shape(_x)[1]
    height = np.shape(_x)[2]
    pool_size = [width, height]
    return tf.layers.average_pooling2d(inputs=_x, pool_size=pool_size, strides=stride)


def batch_normalization(_x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True):
        return tf.cond(training,
                       lambda: batch_norm(inputs=_x, is_training=training, reuse=None),
                       lambda: batch_norm(inputs=_x, is_training=training, reuse=True))


def drop_out(_x, rate, training):
    return tf.layers.dropout(inputs=_x, rate=rate, training=training)


def relu(_x):
    return tf.nn.relu(_x)


def average_pooling(_x, pool_size, stride=2, padding='VALID'):
    return tf.layers.average_pooling2d(inputs=_x, pool_size=pool_size, strides=stride, padding=padding)


def max_pooling(_x, pool_size, stride=2, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=_x, pool_size=pool_size, strides=stride, padding=padding)


def concatenation(layers):
    return tf.concat(layers, axis=3)


def linear(_x, name='linear'):
    return tf.layers.dense(inputs=_x, units=class_num, name=name)


class DenseNet:

    def __init__(self):
        self.nb_blocks = 2
        self.filters = FILTERS
        self.training = training_flag

    def bottleneck_layer(self, _x, scope):

        with tf.name_scope(scope):
            _x = batch_normalization(_x, training=self.training, scope=scope + '_batch1')
            _x = relu(_x)
            _x = conv_layer(_x, _filter=4 * self.filters, kernel=[1, 1], layer_name=scope + '_conv1')
            _x = drop_out(_x, rate=dropout_rate, training=self.training)

            _x = batch_normalization(_x, training=self.training, scope=scope + '_batch2')
            _x = relu(_x)
            _x = conv_layer(_x, _filter=self.filters, kernel=[3, 3], layer_name=scope + '_conv2')
            _x = drop_out(_x, rate=dropout_rate, training=self.training)

            return _x

    def transition_layer(self, _x, scope):
        with tf.name_scope(scope):
            _x = batch_normalization(_x, training=self.training, scope=scope + '_batch1')
            _x = relu(_x)
            _x = conv_layer(_x, _filter=self.filters, kernel=[1, 1], layer_name=scope + '_conv1')
            _x = drop_out(_x, rate=dropout_rate, training=self.training)
            _x = average_pooling(_x, pool_size=[2, 2], stride=2)

            return _x

    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            layers_concat = list()
            layers_concat.append(input_x)

            _x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))

            layers_concat.append(_x)

            for i in range(nb_layers - 1):
                _x = concatenation(layers_concat)
                _x = self.bottleneck_layer(_x, scope=layer_name + '_bottleN_' + str(i + 1))
                layers_concat.append(_x)

            return _x

    def network(self):
        _input = tf.reshape(x, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
        _x = conv_layer(_input, _filter=2 * self.filters, kernel=[7, 7], stride=2, layer_name='conv0')
        _x = max_pooling(_x, pool_size=[3, 3], stride=2)

        for i in range(self.nb_blocks):
            # 6 -> 12 -> 48
            _x = self.dense_block(input_x=_x, nb_layers=16, layer_name='dense_' + str(i))
            _x = self.transition_layer(_x, scope='trans_' + str(i))

        _x = self.dense_block(input_x=_x, nb_layers=32, layer_name='dense_final')

        # 100 Layer
        _x = batch_normalization(_x, training=self.training, scope='linear_batch')
        _x = relu(_x)
        _x = global_average_pooling(_x)
        _x = flatten(_x)
        with tf.name_scope('output'):
            _x = linear(_x)
            final_output = tf.reshape(_x, [-1, MAX_CAPTCHA_LEN, CHAR_SET_LEN])
            predict = tf.argmax(final_output, 1, name='predict')
        return dict(
            predict=predict,
            final_output=final_output
        )
