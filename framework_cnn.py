#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>

import tensorflow as tf
import math
from config import *

w_alpha = 0.01
b_alpha = 0.1
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
# 待卷积的数据 [batch, in_height, in_width, in_channels]
x = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNEL], name='input')
label = tf.placeholder(tf.float32, [None, MAX_CAPTCHA_LEN * CHAR_SET_LEN], name='labels')


class CNN:

    @staticmethod
    def add_conv_layer(pre_conv, pre_neu_num, neu_num, num):
        # filter: [filter_height, filter_width, in_channels, out_channels]

        with tf.name_scope('conv_layer_{}'.format(num)):
            with tf.name_scope('weight'):
                # Patch Size: [CONV_CORE_SIZE*CONV_CORE_SIZE]
                # pre_neu_num: [Number of Input Channels]
                # neu_num: [Number of Output Channels]
                weight = tf.Variable(w_alpha * tf.random_normal([CONV_CORE_SIZE, CONV_CORE_SIZE, pre_neu_num, neu_num]))
            with tf.name_scope('biases'):
                # Before Convolution: ?*IMAGE_HEIGHT * IMAGE_WIDTH * pre_neu_num
                # After Convolution: ?*IMAGE_HEIGHT * IMAGE_WIDTH * neu_num
                biases = tf.Variable(b_alpha * tf.random_normal([neu_num]))
            conv = tf.nn.relu(
                tf.nn.bias_add(tf.nn.conv2d(pre_conv, weight, strides=CONV_STRIDES, padding=PADDING), biases))
            conv = tf.nn.max_pool(conv, ksize=POOL_STRIDES, strides=POOL_STRIDES, padding=PADDING)
            conv = tf.nn.dropout(conv, keep_prob)
            return conv, pre_neu_num, neu_num

    def conv_layer(self, pre_conv, num: int = 1):
        neu_num = CONV_NEU_NUMS[num - 1]
        in_channels = IMAGE_CHANNEL if num == 1 else CONV_NEU_NUMS[num - 2]
        conv, pre_neu_num, neu_num = self.add_conv_layer(pre_conv, in_channels, neu_num, num)
        if num == NEU_LAYER_NUM:
            return conv, neu_num
        return self.conv_layer(conv, num + 1)

    def network(self):
        _x = tf.reshape(x, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])

        conv, neu_num = self.conv_layer(_x)

        # Fully Connected Layer (Dense Layer)
        with tf.name_scope('fc_layer'):
            r = int(
                math.ceil(IMAGE_HEIGHT / (2 ** MAX_POOL_NUM)) * math.ceil(IMAGE_WIDTH / (2 ** MAX_POOL_NUM)) * neu_num)
            with tf.name_scope('weight'):
                weight_d = tf.Variable(w_alpha * tf.random_normal([r, FULL_LAYER_FEATURE_NUM]))
            with tf.name_scope('biases'):
                biases_d = tf.Variable(b_alpha * tf.random_normal([FULL_LAYER_FEATURE_NUM]))
            dense = tf.reshape(conv, [-1, weight_d.get_shape().as_list()[0]])
            dense = tf.nn.relu(tf.add(tf.matmul(dense, weight_d), biases_d))
            dense = tf.nn.dropout(dense, keep_prob)
        with tf.name_scope('output'):
            with tf.name_scope('weight'):
                weight_out = tf.Variable(
                    w_alpha * tf.random_normal([FULL_LAYER_FEATURE_NUM, MAX_CAPTCHA_LEN * CHAR_SET_LEN]))
            with tf.name_scope('biases'):
                biases_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA_LEN * CHAR_SET_LEN]))
            output = tf.add(tf.matmul(dense, weight_out), biases_out)
            final_output = tf.reshape(output, [-1, MAX_CAPTCHA_LEN, CHAR_SET_LEN])
            predict = tf.argmax(final_output, 2, name='predict')
        return dict(
            predict=predict,
            final_output=final_output
        )
