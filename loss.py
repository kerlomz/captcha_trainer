#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import tensorflow as tf
from config import ModelConfig


class Loss(object):

    """损失函数生成器"""
    @staticmethod
    def cross_entropy(labels, logits):
        """交叉熵损失函数"""

        # return tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
        # return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        # return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        # return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        target = tf.sparse.to_dense(labels)
        # target = labels
        print('logits', logits.shape)
        print('target', target.shape)
        # logits = tf.reshape(tensor=logits, shape=[tf.shape(labels)[0], None])
        return tf.keras.backend.sparse_categorical_crossentropy(
            target=target,
            output=logits,
            from_logits=True,
        )

    @staticmethod
    def ctc(labels, logits, sequence_length):
        """CTC 损失函数"""

        return tf.compat.v1.nn.ctc_loss_v2(
            labels=labels,
            logits=logits,
            logit_length=sequence_length,
            label_length=sequence_length,
            blank_index=-1,
            logits_time_major=True
        )
