#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import sys
import itertools
import tensorflow as tf
from config import *
from network.CNN import CNN5, CNNX
from network.DenseNet import DenseNet
from network.GRU import GRU, BiGRU, GRUcuDNN
from network.LSTM import LSTM, BiLSTM, BiLSTMcuDNN, LSTMcuDNN
from network.ResNet import ResNet50
from network.utils import NetworkUtils
from optimizer.AdaBound import AdaBoundOptimizer
from tensorflow.python.keras.regularizers import *

slim = tf.contrib.slim


class GraphOCR(object):

    def __init__(self, mode: RunMode, cnn: CNNNetwork, recurrent: RecurrentNetwork):
        self.mode = mode
        self.utils = NetworkUtils(mode)
        self.network = cnn
        self.recurrent = recurrent
        self.inputs = tf.keras.Input(dtype=tf.float32, shape=[None, RESIZE[1], IMAGE_CHANNEL], name='input')
        self.labels = tf.keras.Input(dtype=tf.int32, shape=[None], sparse=True, name='labels')
        self.seq_len = None
        self.logits = None
        self.merged_summary = None

    def build_graph(self):
        self._build_model()
        self._build_train_op()
        self.merged_summary = tf.compat.v1.summary.merge_all()

    def _build_model(self):

        if self.network == CNNNetwork.CNN5:
            x = CNN5(inputs=self.inputs, utils=self.utils).build()

        elif self.network == CNNNetwork.CNNX:
            x = CNNX(inputs=self.inputs, utils=self.utils).build()

        elif self.network == CNNNetwork.ResNet:
            x = ResNet50(inputs=self.inputs, utils=self.utils).build()

        # This network was temporarily suspended
        elif self.network == CNNNetwork.DenseNet:
            x = DenseNet(inputs=self.inputs, utils=self.utils).build()

        else:
            tf.logging.error('This cnn neural network is not supported at this time.')
            sys.exit(-1)

        # time_major = True: [max_time_step, batch_size, num_classes]
        # time_major = False: [batch_size, max_time_step, num_classes]
        tf.compat.v1.logging.info("CNN Output: {}".format(x.get_shape()))

        self.seq_len = tf.fill([tf.shape(x)[0]], tf.shape(x)[1], name="seq_len")

        if self.recurrent == RecurrentNetwork.LSTM:
            self.recurrent_network_builder = LSTM(x, utils=self.utils)
        elif self.recurrent == RecurrentNetwork.BiLSTM:
            self.recurrent_network_builder = BiLSTM(x, utils=self.utils)
        elif self.recurrent == RecurrentNetwork.GRU:
            self.recurrent_network_builder = GRU(x, utils=self.utils)
        elif self.recurrent == RecurrentNetwork.BiGRU:
            self.recurrent_network_builder = BiGRU(x, utils=self.utils)
        elif self.recurrent == RecurrentNetwork.LSTMcuDNN:
            self.recurrent_network_builder = LSTMcuDNN(x, utils=self.utils)
        elif self.recurrent == RecurrentNetwork.BiLSTMcuDNN:
            self.recurrent_network_builder = BiLSTMcuDNN(x, utils=self.utils)
        elif self.recurrent == RecurrentNetwork.GRUcuDNN:
            self.recurrent_network_builder = GRUcuDNN(x, utils=self.utils)
        else:
            tf.logging.error('This recurrent neural network is not supported at this time.')
            sys.exit(-1)

        outputs = self.recurrent_network_builder.build()
        self.outputs = outputs

        with tf.variable_scope('output'):

            self.logits = tf.keras.layers.Dense(
                units=NUM_CLASSES,
                kernel_initializer=tf.keras.initializers.glorot_normal(seed=None),
                kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
                bias_initializer='zeros',
            )
            predict = tf.keras.layers.TimeDistributed(
                layer=self.logits,
                name='predict',
                trainable=self.utils.training
            )(inputs=outputs, training=self.utils.training)

            self.predict = tf.transpose(predict, perm=(1, 0, 2))

    def _build_train_op(self):
        self.global_step = tf.train.get_or_create_global_step()

        self.loss = tf.compat.v1.nn.ctc_loss(
            labels=self.labels,
            inputs=self.predict,
            sequence_length=self.seq_len,
            ctc_merge_repeated=CTC_MERGE_REPEATED,
            preprocess_collapse_repeated=PREPROCESS_COLLAPSE_REPEATED,
            ignore_longer_outputs_than_inputs=False,
            time_major=True
        )

        self.cost = tf.reduce_mean(self.loss)
        tf.compat.v1.summary.scalar('cost', self.cost)
        self.lrn_rate = tf.compat.v1.train.exponential_decay(
            TRAINS_LEARNING_RATE,
            self.global_step,
            DECAY_STEPS,
            DECAY_RATE,
            staircase=True
        )
        tf.compat.v1.summary.scalar('learning_rate', self.lrn_rate)

        update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)

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

        self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(
            inputs=self.predict,
            sequence_length=self.seq_len,
            beam_width=CTC_BEAM_WIDTH,
            top_paths=CTC_TOP_PATHS,
        )

        self.dense_decoded = tf.sparse.to_dense(self.decoded[0], default_value=-1, name="dense_decoded")


if __name__ == '__main__':
    GraphOCR(RunMode.Trains, CNNNetwork.CNN5, RecurrentNetwork.GRU).build_graph()

