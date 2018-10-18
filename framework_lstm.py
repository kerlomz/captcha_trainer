#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import tensorflow as tf
from config import *

class LSTM(object):

    def __init__(self, mode):
        self.mode = mode
        self.batch_size = tf.placeholder(tf.int32, name="batch_size")
        self.inputs = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, 1], name='input')
        self.labels = tf.sparse_placeholder(tf.int32, name='labels')
        self._extra_train_ops = []
        self.merged_summary = None

    def build_graph(self):
        self._build_model()
        self._build_train_op()
        self.merged_summary = tf.summary.merge_all()

    def _build_model(self):
        feature_w, feature_h = IMAGE_WIDTH, IMAGE_HEIGHT
        max_cnn_layer_num = 0
        min_size = min(IMAGE_HEIGHT, IMAGE_WIDTH)
        while min_size > 1:
            min_size = (min_size + 1) // 2
            max_cnn_layer_num += 1
        assert (len(CNN_STRUCTURE) <= max_cnn_layer_num, "CNN_NEU_STRUCTURE should be less than {}!".format(max_cnn_layer_num))

        with tf.variable_scope('cnn'):
            x = self.inputs
            for i, neu in enumerate(CNN_STRUCTURE):
                with tf.variable_scope('unit-%d' % (i + 1)):
                    x = self._conv2d(x, 'cnn-%d' % (i + 1), CONV_KSIZE[i], FILTERS[i], FILTERS[i + 1], CONV_STRIDES[i])
                    x = self._batch_norm('bn%d' % (i + 1), x)
                    x = self._leaky_relu(x, LEAKINESS)
                    x = self._max_pool(x, POOL_KSIZE[i], POOL_STRIDES[i])
                    _, feature_h, feature_w, _ = x.get_shape().as_list()

        with tf.variable_scope('lstm'):
            x = tf.transpose(x, perm=[0, 2, 1, 3])
            # Treat `feature_w` as max_time_step in lstm.
            x = tf.reshape(x, [self.batch_size, feature_w, feature_h * OUT_CHANNEL])

            self.seq_len = tf.fill([self.batch_size], feature_w, name="seq_len")

            cell = tf.nn.rnn_cell.LSTMCell(NUM_HIDDEN, state_is_tuple=True)
            if self.mode == RunMode.Trains:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=OUTPUT_KEEP_PROB)

            cell1 = tf.nn.rnn_cell.LSTMCell(NUM_HIDDEN, state_is_tuple=True)
            if self.mode == RunMode.Trains:
                cell1 = tf.nn.rnn_cell.DropoutWrapper(cell=cell1, output_keep_prob=OUTPUT_KEEP_PROB)

            # Stacking rnn cells
            stack = tf.nn.rnn_cell.MultiRNNCell([cell, cell1], state_is_tuple=True)

            initial_state = stack.zero_state(tf.shape(self.inputs)[0], tf.float32)

            outputs, _ = tf.nn.dynamic_rnn(
                cell=stack,
                inputs=x,
                sequence_length=self.seq_len,
                initial_state=initial_state,
                dtype=tf.float32,
                time_major=False,
                scope="rnn_output"
            )

            # Reshaping to apply the same weights over the time_steps
            outputs = tf.reshape(outputs, [-1, NUM_HIDDEN])
            with tf.variable_scope('output'):
                weight_out = tf.get_variable(
                    name='weight',
                    shape=[NUM_HIDDEN, NUM_CLASSES],
                    dtype=tf.float32,
                    initializer=tf.glorot_uniform_initializer()
                )
                biases_out = tf.get_variable(
                    name='biases',
                    shape=[NUM_CLASSES],
                    dtype=tf.float32,
                    initializer=tf.constant_initializer()
                )
                output = tf.add(tf.matmul(outputs, weight_out), biases_out)
                output = tf.reshape(output, [tf.shape(x)[0], -1, NUM_CLASSES])
                predict = tf.transpose(output, (1, 0, 2), "predict")
                self.predict = predict

    def _build_train_op(self):

        self.global_step = tf.train.get_or_create_global_step()
        self.loss = tf.nn.ctc_loss(
            labels=self.labels,
            inputs=self.predict,
            sequence_length=self.seq_len,
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

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.lrn_rate,
            beta1=BATE1,
            beta2=BATE2
        ).minimize(
            self.loss,
            global_step=self.global_step
        )
        train_ops = [self.optimizer] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)

        self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(
            self.predict,
            self.seq_len,
            merge_repeated=False,
        )
        self.dense_decoded = tf.sparse_tensor_to_dense(self.decoded[0], default_value=-1)

    @staticmethod
    def _conv2d(x, name, filter_size, in_channels, out_channels, strides):
        with tf.variable_scope(name):
            kernel = tf.get_variable(
                name='weight',
                shape=[filter_size, filter_size, in_channels, out_channels],
                dtype=tf.float32,
                initializer=tf.glorot_uniform_initializer())

            biases = tf.get_variable(
                name='biases',
                shape=[out_channels],
                dtype=tf.float32,
                initializer=tf.constant_initializer()
            )

            con2d_op = tf.nn.conv2d(x, kernel, [1, strides, strides, 1], padding='SAME')

        return tf.nn.bias_add(con2d_op, biases)

    def _batch_norm(self, name, x):
        """Batch normalization."""
        with tf.variable_scope(name):
            x_bn = tf.contrib.layers.batch_norm(
                inputs=x,
                decay=0.9,
                center=True,
                scale=True,
                epsilon=1e-5,
                updates_collections=None,
                is_training=self.mode == RunMode.Trains,
                fused=True,
                data_format='NHWC',
                zero_debias_moving_mean=True,
                scope='BatchNorm'
            )

        return x_bn

    @staticmethod
    def _leaky_relu(x, leakiness=0.0):
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    @staticmethod
    def _max_pool(x, ksize, strides):
        return tf.nn.max_pool(
            x,
            ksize=[1, ksize, ksize, 1],
            strides=[1, strides, strides, 1],
            padding='SAME',
            name='max_pool'
        )
