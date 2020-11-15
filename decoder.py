#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import tensorflow as tf
from config import ModelConfig


class Decoder:
    """
    转录层：用于解码预测结果
    """
    def __init__(self, model_conf: ModelConfig):
        self.model_conf = model_conf
        self.category_num = self.model_conf.category_num

    def ctc(self, inputs, sequence_length):
        """针对CTC Loss的解码"""
        ctc_decode, _ = tf.compat.v1.nn.ctc_beam_search_decoder_v2(inputs, sequence_length, beam_width=1)
        decoded_sequences = tf.sparse.to_dense(ctc_decode[0], default_value=self.category_num, name='dense_decoded')
        return decoded_sequences

    @staticmethod
    def cross_entropy(inputs):
        """针对CrossEntropy Loss的解码"""
        return tf.argmax(inputs, 2, name='dense_decoded')

