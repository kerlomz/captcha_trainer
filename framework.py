#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import sys
import tensorflow as tf
from config import *
from network.utils import NetworkUtils
from network.CNN5 import CNN5
from network.ResNet import ResNet50
from network.LSTM import LSTM, BLSTM
from network.SRU import SRU, BSRU
from network.GRU import GRU


class GraphOCR(object):

    def __init__(self, mode, cnn: CNNNetwork, recurrent: RecurrentNetwork):
        self.mode = mode
        self.utils = NetworkUtils(mode)
        self.network = cnn
        self.recurrent = recurrent
        self.inputs = tf.placeholder(tf.float32, [None, RESIZE[0], RESIZE[1], 1], name='input')
        self.labels = tf.sparse_placeholder(tf.int32, name='labels')
        self._extra_train_ops = []
        self.seq_len = None
        self.merged_summary = None

    def build_graph(self):
        self._build_model()
        self._build_train_op()
        self.merged_summary = tf.summary.merge_all()

    def _build_model(self):
        if self.network == CNNNetwork.CNN5:
            x = CNN5(inputs=self.inputs, utils=self.utils).build()

        elif self.network == CNNNetwork.ResNet:
            x = ResNet50(inputs=self.inputs, utils=self.utils).build()

        else:
            print('This cnn neural network is not supported at this time.')
            sys.exit(-1)

        shape_list = x.get_shape().as_list()
        self.seq_len = tf.fill([tf.shape(x)[0]], shape_list[1], name="seq_len")
        if self.recurrent == RecurrentNetwork.LSTM:
            outputs = LSTM(self.mode, x, self.seq_len).build()

        elif self.recurrent == RecurrentNetwork.BLSTM:
            outputs = BLSTM(self.mode, self.utils, x, self.seq_len).build()
        elif self.recurrent == RecurrentNetwork.GRU:
            outputs = GRU(self.inputs, self.seq_len).build()

        elif self.recurrent == RecurrentNetwork.SRU:
            outputs = SRU(x, self.seq_len).build()

        elif self.recurrent == RecurrentNetwork.BSRU:
            outputs = BSRU(self.utils, x, self.seq_len)
        else:
            print('This recurrent neural network is not supported at this time.')
            sys.exit(-1)

        # Reshaping to apply the same weights over the time_steps
        outputs = tf.reshape(outputs, [-1, NUM_HIDDEN * 2])
        with tf.variable_scope('output'):
            # tf.Variable
            weight_out = tf.get_variable(
                name='weight',
                shape=[outputs.get_shape()[1] if self.network == CNNNetwork.ResNet else NUM_HIDDEN * 2, NUM_CLASSES],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=0.1),
                # initializer=tf.glorot_uniform_initializer(),
                # initializer=tf.contrib.layers.xavier_initializer(),
                # initializer=tf.truncated_normal([NUM_HIDDEN, NUM_CLASSES], stddev=0.1),
            )
            biases_out = tf.get_variable(
                name='biases',
                shape=[NUM_CLASSES],
                dtype=tf.float32,
                initializer=tf.constant_initializer(value=0, dtype=tf.float32)
            )
            # [batch_size * max_timesteps, num_classes]
            logits = tf.matmul(outputs, weight_out) + biases_out
            # Reshaping back to the original shape
            logits = tf.reshape(logits, [tf.shape(x)[0], -1, NUM_CLASSES])
            # Time major
            predict = tf.transpose(logits, (1, 0, 2), "predict")
            self.predict = predict

    def _build_train_op(self):
        self.global_step = tf.train.get_or_create_global_step()
        # ctc loss function, using forward and backward algorithms and maximum likelihood.

        self.loss = tf.nn.ctc_loss(
            labels=self.labels,
            inputs=self.predict,
            sequence_length=self.seq_len,
            ctc_merge_repeated=CTC_MERGE_REPEATED,
            preprocess_collapse_repeated=PREPROCESS_COLLAPSE_REPEATED,
            ignore_longer_outputs_than_inputs=False,
            time_major=True
        )

        self.cost = tf.reduce_mean(self.loss)
        tf.summary.scalar('cost', self.cost)

        self.lrn_rate = tf.train.exponential_decay(
            TRAINS_LEARNING_RATE,
            self.global_step,
            DECAY_STEPS,
            DECAY_RATE,
            staircase=True
        )
        tf.summary.scalar('learning_rate', self.lrn_rate)

        self.optimizer = tf.train.MomentumOptimizer(
            learning_rate=self.lrn_rate,
            use_nesterov=True,
            momentum=MOMENTUM,
        ).minimize(
            self.cost,
            global_step=self.global_step
        )

        # Storing adjusted smoothed mean and smoothed variance operations
        train_ops = [self.optimizer] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)

        # Option 2: tf.contrib.ctc.ctc_beam_search_decoder
        # (it's slower but you'll get better results)
        # self.decoded, self.log_prob = tf.nn.ctc_greedy_decoder(
        #     self.predict,
        #     self.seq_len,
        #     merge_repeated=False
        # )

        # Find the optimal path
        self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(
            self.predict,
            self.seq_len,
            merge_repeated=False,
            beam_width=CTC_BEAM_WIDTH,
            top_paths=CTC_TOP_PATHS,
        )

        self.dense_decoded = tf.sparse.to_dense(self.decoded[0], default_value=-1, name="dense_decoded")

