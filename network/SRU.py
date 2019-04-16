#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import RNNCell
from config import NUM_HIDDEN, LSTM_LAYER_NUM
from network.utils import NetworkUtils


class SRU(object):

    def __init__(self, inputs: tf.Tensor, seq_len: tf.Tensor):
        self.inputs = inputs
        self.seq_len = seq_len

    def build(self):
        with tf.variable_scope('SRU'):
            cell1 = SRUCell(NUM_HIDDEN * 2, False)
            cell2 = SRUCell(NUM_HIDDEN * 2, False)

            stack = tf.contrib.rnn.MultiRNNCell([cell1, cell2], state_is_tuple=True)
            outputs, _ = tf.nn.dynamic_rnn(stack, self.inputs, self.seq_len, dtype=tf.float32)
        return outputs


class BSRU(object):

    def __init__(self, utils: NetworkUtils, inputs: tf.Tensor, seq_len: tf.Tensor):
        self.utils = utils
        self.inputs = inputs
        self.seq_len = seq_len

    def build(self):
        with tf.variable_scope('BSRU'):
            outputs = self.utils.stacked_bidirectional_rnn(
                SRUCell,
                NUM_HIDDEN,
                LSTM_LAYER_NUM,
                self.inputs,
                self.seq_len
            )
        return outputs


class SRUCell(RNNCell):

    def __init__(self, num_units, using_highway=True):
        super(SRUCell, self).__init__()
        self._num_units = num_units
        self._using_highway = using_highway

    @property
    def state_size(self):
        return self._num_units, self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, x, state, scope=None):
        if self._using_highway:
            return self.call_with_highway(x, state, scope)
        else:
            return self.call_without_highway(x, state, scope)

    def call_without_highway(self, x, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            c, _ = state
            x_size = x.get_shape().as_list()[1]

            w_u = tf.get_variable('W_u', [x_size, 3 * self.output_size])
            b_f = tf.get_variable('b_f', [self._num_units])
            b_r = tf.get_variable('b_r', [self._num_units])

            xh = tf.matmul(x, w_u)
            z, f, r = tf.split(xh, 3, 1)

            f = tf.sigmoid(f + b_f)
            r = tf.sigmoid(r + b_r)

            new_c = f * c + (1 - f) * z
            new_h = r * tf.tanh(new_c)

            return new_h, (new_c, new_h)

    def call_with_highway(self, x, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            c, _ = state
            x_size = x.get_shape().as_list()[1]

            w_u = tf.get_variable('W_u', [x_size, 4 * self.output_size])
            b_f = tf.get_variable('b_f', [self._num_units])
            b_r = tf.get_variable('b_r', [self._num_units])

            xh = tf.matmul(x, w_u)
            z, f, r, x = tf.split(xh, 4, 1)

            f = tf.sigmoid(f + b_f)
            r = tf.sigmoid(r + b_r)

            new_c = f * c + (1 - f) * z
            new_h = r * tf.tanh(new_c) + (1 - r) * x

            return new_h, (new_c, new_h)



