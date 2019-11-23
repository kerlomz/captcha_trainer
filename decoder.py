#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import tensorflow as tf
from config import ModelConfig


class Decoder:

    def __init__(self, model_conf: ModelConfig):
        self.model_conf = model_conf
        self.category_num = self.model_conf.category_num

    def ctc(self, inputs, sequence_length):
        ctc_decode, _ = tf.nn.ctc_greedy_decoder(inputs, sequence_length)
        decoded_sequences = tf.sparse.to_dense(ctc_decode[0], default_value=self.category_num, name='dense_decoded')
        return decoded_sequences

    @staticmethod
    def cross_entropy(inputs):
        return tf.argmax(inputs, 2, name='dense_decoded')

