#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import tensorflow as tf
from config import NUM_HIDDEN, LSTM_LAYER_NUM, RunMode
from network.utils import NetworkUtils


class LSTM(object):

    def __init__(self, utils: NetworkUtils, inputs: tf.Tensor, seq_len: tf.Tensor):
        self.mode = utils.mode
        self.inputs = inputs
        self.seq_len = seq_len

    def build(self):
        with tf.variable_scope('LSTM'):

            cell1 = tf.contrib.rnn.LSTMCell(NUM_HIDDEN * 2, state_is_tuple=True)
            if self.mode == RunMode.Trains:
                cell1 = tf.contrib.rnn.DropoutWrapper(cell=cell1, output_keep_prob=0.8)
            cell2 = tf.contrib.rnn.LSTMCell(NUM_HIDDEN * 2, state_is_tuple=True)
            if self.mode == RunMode.Trains:
                cell2 = tf.contrib.rnn.DropoutWrapper(cell=cell2, output_keep_prob=0.8)

            stack = tf.contrib.rnn.MultiRNNCell([cell1, cell2], state_is_tuple=True)
            outputs, _ = tf.nn.dynamic_rnn(stack, self.inputs, self.seq_len, dtype=tf.float32)

        return outputs


class BLSTM(object):

    def __init__(self, utils: NetworkUtils, inputs: tf.Tensor, seq_len: tf.Tensor):
        self.utils = utils
        self.inputs = inputs
        self.seq_len = seq_len

    def build(self):
        with tf.variable_scope('BLSTM'):
            outputs = self.utils.stacked_bidirectional_rnn(
                tf.contrib.rnn.LSTMCell,
                NUM_HIDDEN,
                LSTM_LAYER_NUM,
                self.inputs,
                self.seq_len
            )
        return outputs
