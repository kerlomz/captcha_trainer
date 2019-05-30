#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import sys
import tensorflow as tf
from importlib import import_module
from distutils.version import StrictVersion
from config import *
from network.CNN import CNN5
from network.GRU import GRU
from network.LSTM import LSTM, BLSTM
from network.ResNet import ResNet50
from network.DenseNet import DenseNet
from network.SRU import SRU, BSRU
from network.utils import NetworkUtils
from optimizer.AdaBound import AdaBoundOptimizer


class GraphOCR(object):

    def __init__(self, mode, cnn: CNNNetwork, recurrent: RecurrentNetwork):
        self.mode = mode
        self.utils = NetworkUtils(mode)
        self.network = cnn
        self.recurrent = recurrent
        self.inputs = tf.placeholder(tf.float32, [None, None, RESIZE[1], IMAGE_CHANNEL], name='input')
        self.labels = tf.sparse_placeholder(tf.int32, name='labels')
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

        elif self.network == CNNNetwork.DenseNet:
            x = DenseNet(inputs=self.inputs, utils=self.utils).build()

        else:
            tf.logging.error('This cnn neural network is not supported at this time.')
            sys.exit(-1)

        # time_major = True: [max_time_step, batch_size, num_classes]
        # time_major = False: [batch_size, max_time_step, num_classes]
        tf.logging.info("CNN Output: {}".format(x.get_shape()))

        self.seq_len = tf.fill([tf.shape(x)[0]], tf.shape(x)[1], name="seq_len")

        if self.recurrent == RecurrentNetwork.LSTM:
            recurrent_network_builder = LSTM(self.utils, x, self.seq_len)
        elif self.recurrent == RecurrentNetwork.BLSTM:
            recurrent_network_builder = BLSTM(self.utils, x, self.seq_len)
        elif self.recurrent == RecurrentNetwork.GRU:
            recurrent_network_builder = GRU(x, self.seq_len)
        elif self.recurrent == RecurrentNetwork.SRU:
            recurrent_network_builder = SRU(x, self.seq_len)
        elif self.recurrent == RecurrentNetwork.BSRU:
            recurrent_network_builder = BSRU(self.utils, x, self.seq_len)
        else:
            tf.logging.error('This recurrent neural network is not supported at this time.')
            sys.exit(-1)

        outputs = recurrent_network_builder.build()

        # Reshaping to apply the same weights over the time_steps
        outputs = tf.reshape(outputs, [-1, NUM_HIDDEN * 2])
        with tf.variable_scope('output'):
            # tf.Variable
            weight_out = tf.get_variable(
                name='weight',
                shape=[outputs.get_shape()[1] if self.network == CNNNetwork.ResNet else NUM_HIDDEN * 2, NUM_CLASSES],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer(),
                # initializer=tf.truncated_normal_initializer(stddev=0.1),
                # initializer=tf.glorot_normal_initializer(),
                # initializer=tf.glorot_uniform_initializer(),
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
        if WARP_CTC:
            import_module('warpctc_tensorflow')
            with tf.get_default_graph()._kernel_label_map({"CTCLoss": "WarpCTC"}):
                self.loss = tf.nn.ctc_loss(
                    inputs=self.predict,
                    labels=self.labels,
                    sequence_length=self.seq_len
                )
        else:
            self.loss = tf.nn.ctc_loss(
                labels=self.labels,
                inputs=self.predict,
                sequence_length=self.seq_len,
                ctc_merge_repeated=CTC_MERGE_REPEATED,
                preprocess_collapse_repeated=PREPROCESS_COLLAPSE_REPEATED,
                ignore_longer_outputs_than_inputs=False,
                time_major=CTC_LOSS_TIME_MAJOR
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

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # print(update_ops)
        # Storing adjusted smoothed mean and smoothed variance operations
        with tf.control_dependencies(update_ops):
            if OPTIMIZER_MAP[NEU_OPTIMIZER] == Optimizer.AdaBound:
                self.train_op = AdaBoundOptimizer(
                    learning_rate=self.lrn_rate,
                    final_lr=0.1,
                    beta1=0.9,
                    beta2=0.999,
                    amsbound=True
                ).minimize(
                    loss=self.cost,
                    global_step=self.global_step
                )
            elif OPTIMIZER_MAP[NEU_OPTIMIZER] == Optimizer.Adam:
                self.train_op = tf.train.AdamOptimizer(
                    learning_rate=self.lrn_rate
                ).minimize(
                    self.cost,
                    global_step=self.global_step
                )
            elif OPTIMIZER_MAP[NEU_OPTIMIZER] == Optimizer.Momentum:
                self.train_op = tf.train.MomentumOptimizer(
                    learning_rate=self.lrn_rate,
                    use_nesterov=True,
                    momentum=MOMENTUM,
                ).minimize(
                    self.cost,
                    global_step=self.global_step
                )
            elif OPTIMIZER_MAP[NEU_OPTIMIZER] == Optimizer.SGD:
                self.train_op = tf.train.GradientDescentOptimizer(
                    learning_rate=self.lrn_rate,
                ).minimize(
                    self.cost,
                    global_step=self.global_step
                )
            elif OPTIMIZER_MAP[NEU_OPTIMIZER] == Optimizer.AdaGrad:
                self.train_op = tf.train.AdagradOptimizer(
                    learning_rate=self.lrn_rate,
                ).minimize(
                    self.cost,
                    global_step=self.global_step
                )
            elif OPTIMIZER_MAP[NEU_OPTIMIZER] == Optimizer.RMSProp:
                self.train_op = tf.train.RMSPropOptimizer(
                    learning_rate=self.lrn_rate,
                    decay=DECAY_RATE,
                ).minimize(
                    self.cost,
                    global_step=self.global_step
                )

        # Option 2: tf.contrib.ctc.ctc_beam_search_decoder
        # (it's slower but you'll get better results)
        # self.decoded, self.log_prob = tf.nn.ctc_greedy_decoder(
        #     self.predict,
        #     self.seq_len,
        #     merge_repeated=False
        # )

        # Find the optimal path
        self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(
            inputs=self.predict,
            sequence_length=self.seq_len,
            merge_repeated=False,
            beam_width=CTC_BEAM_WIDTH,
            top_paths=CTC_TOP_PATHS,
        )

        if StrictVersion(tf.__version__) >= StrictVersion('1.12.0'):
            self.dense_decoded = tf.sparse.to_dense(self.decoded[0], default_value=-1, name="dense_decoded")
        else:
            self.dense_decoded = tf.sparse_tensor_to_dense(self.decoded[0], default_value=-1, name="dense_decoded")


if __name__ == '__main__':
    GraphOCR(RunMode.Predict, CNNNetwork.CNN5, RecurrentNetwork.BLSTM).build_graph()
