#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>

import tensorflow as tf
from config import NUM_HIDDEN


class GRU(object):

    def __init__(self, inputs: tf.Tensor, seq_len: tf.Tensor):
        self.inputs = inputs
        self.seq_len = seq_len

    def build(self):
        with tf.variable_scope('GRU'):
            cell = tf.nn.rnn_cell.GRUCell(NUM_HIDDEN * 2)
            outputs, _ = tf.nn.dynamic_rnn(cell, self.inputs, self.seq_len, dtype=tf.float32)
        return outputs
