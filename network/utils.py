#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import math
import tensorflow as tf
from config import *


class NetworkUtils(object):

    def __init__(self, mode):
        self.extra_train_ops = []
        self.mode = mode

    @staticmethod
    def zero_padding(x, pad=(3, 3)):
        padding = tf.constant([[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]])
        return tf.pad(x, padding, 'CONSTANT')

    @staticmethod
    def msra_initializer(kl, dl):
        """ MSRA weight initializer
        (https://arxiv.org/pdf/1502.01852.pdf)
        Keyword arguments:
        kl -- kernel size
        dl -- filter numbers
        """

        stddev = math.sqrt(2. / (kl ** 2 * dl))
        return tf.truncated_normal_initializer(stddev=stddev)

    def cnn_layers(self, inputs, filter_size, filters, strides):
        x = inputs
        for i in range(len(filter_size)):
            with tf.variable_scope('unit-{}'.format(i + 1)):
                x = self.conv2d(
                    x=x,
                    name='cnn-{}'.format(i + 1),
                    filter_size=filter_size[i],
                    in_channels=filters[i][0],
                    out_channels=filters[i][1],
                    strides=strides[i][0]
                )
                x = self.batch_norm('bn{}'.format(i + 1), x)
                x = self.leaky_relu(x, 0.01)
                x = self.max_pool(x, 2, strides[i][1])
        return x

    def conv2d(self, x, name, filter_size, in_channels, out_channels, strides, padding='SAME'):
        # n = filter_size * filter_size * out_channels
        with tf.variable_scope(name):
            kernel = tf.get_variable(
                name='DW',
                shape=[filter_size, filter_size, in_channels, out_channels],
                dtype=tf.float32,
                # initializer=tf.contrib.layers.xavier_initializer(),
                initializer=self.msra_initializer(filter_size, in_channels),
            )

            b = tf.get_variable(
                name='bais',
                shape=[out_channels],
                dtype=tf.float32,
                initializer=tf.constant_initializer()
            )
            con2d_op = tf.nn.conv2d(x, kernel, [1, strides, strides, 1], padding=padding)
        # return con2d_op
        return tf.nn.bias_add(con2d_op, b)

    def identity_block(self, x, f, out_channels, stage, block):
        """
        Implementing a ResNet identity block with shortcut path
        passing over 3 Conv Layers

        @params
        X - input tensor of shape (m, in_H, in_W, in_C)
        f - size of middle layer filter
        out_channels - tuple of number of filters in 3 layers
        stage - used to name the layers
        block - used to name the layers

        @returns
        A - Output of identity_block
        params - Params used in identity block
        """

        conv_name = 'res' + str(stage) + block + '_branch'
        bn_name = 'bn' + str(stage) + block + '_branch'

        input_tensor = x

        _, _, _, in_channels = x.shape.as_list()

        x = self.conv2d(
            x=x,
            name="{}2a".format(conv_name),
            filter_size=1,
            in_channels=in_channels,
            out_channels=out_channels[0],
            strides=1,
            padding='VALID'
        )

        x = self.batch_norm(x=x, name=bn_name + '2a')
        # x = tf.nn.relu(x)
        x = self.leaky_relu(x)

        _, _, _, in_channels = x.shape.as_list()
        x = self.conv2d(
            x=x,
            name="{}2b".format(conv_name),
            filter_size=f,
            in_channels=in_channels,
            out_channels=out_channels[1],
            strides=1,
            padding='SAME'
        )
        x = self.batch_norm(x=x, name=bn_name + '2b')
        # x = tf.nn.relu(x)
        x = self.leaky_relu(x)

        _, _, _, in_channels = x.shape.as_list()
        x = self.conv2d(
            x=x,
            name="{}2c".format(conv_name),
            filter_size=1,
            in_channels=in_channels,
            out_channels=out_channels[2],
            strides=1,
            padding='VALID'
        )
        x = self.batch_norm(x=x, name=bn_name + '2c')

        x = tf.add(input_tensor, x)
        # x = tf.nn.relu(x)
        x = self.leaky_relu(x)

        return x

    def convolutional_block(self, x, f, out_channels, stage, block, s=2):
        """
        Implementing a ResNet convolutional block with shortcut path
        passing over 3 Conv Layers having different sizes

        @params
        X - input tensor of shape (m, in_H, in_W, in_C)
        f - size of middle layer filter
        out_channels - tuple of number of filters in 3 layers
        stage - used to name the layers
        block - used to name the layers
        s - strides used in first layer of convolutional block

        @returns
        A - Output of convolutional_block
        params - Params used in convolutional block
        """

        conv_name = 'res' + str(stage) + block + '_branch'
        bn_name = 'bn' + str(stage) + block + '_branch'

        _, _, _, in_channels = x.shape.as_list()
        a1 = self.conv2d(
            x=x,
            name="{}2a".format(conv_name),
            filter_size=1,
            in_channels=in_channels,
            out_channels=out_channels[0],
            strides=s,
            padding='VALID'
        )
        a1 = self.batch_norm(x=a1, name=bn_name + '2a')
        # a1 = tf.nn.relu(a1)
        a1 = self.leaky_relu(a1)

        _, _, _, in_channels = a1.shape.as_list()
        a2 = self.conv2d(
            x=a1,
            name="{}2b".format(conv_name),
            filter_size=f,
            in_channels=in_channels,
            out_channels=out_channels[1],
            strides=1,
            padding='SAME'
        )
        a2 = self.batch_norm(x=a2, name=bn_name + '2b')
        # a2 = tf.nn.relu(a2)
        a2 = self.leaky_relu(a2)

        _, _, _, in_channels = a2.shape.as_list()
        a3 = self.conv2d(
            x=a2,
            name="{}2c".format(conv_name),
            filter_size=1,
            in_channels=in_channels,
            out_channels=out_channels[2],
            strides=1,
            padding='VALID'
        )
        a3 = self.batch_norm(x=a3, name=bn_name + '2c')
        # a3 = tf.nn.relu(a3)
        a3 = self.leaky_relu(a3)

        _, _, _, in_channels = x.shape.as_list()
        x = self.conv2d(
            x=x,
            name="{}1".format(conv_name),
            filter_size=1,
            in_channels=in_channels,
            out_channels=out_channels[2],
            strides=s,
            padding='VALID'
        )

        x = self.batch_norm(x=x, name=bn_name + '1')

        x = tf.add(a3, x)
        x = self.leaky_relu(x)
        # x = tf.nn.relu(x)

        return x

    # Variant Relu
    # The gradient of the non-negative interval is constant,
    # - which can prevent the gradient from disappearing to some extent.
    @staticmethod
    def leaky_relu(x, leakiness=0.0):
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    @staticmethod
    def max_pool(x, ksize, strides):
        if isinstance(ksize, int):
            ksize = [ksize, ksize]
        if isinstance(strides, int):
            strides = [strides, strides]
        return tf.nn.max_pool(
            x,
            ksize=[1, ksize[0], ksize[1], 1],
            strides=[1, strides[0], strides[1], 1],
            padding='SAME',
            name='max_pool'
        )

    @staticmethod
    def stacked_bidirectional_rnn(rnn, num_units, num_layers, inputs, seq_lengths):
        """
        multi layer bidirectional rnn
        :param rnn: RNN class, e.g. LSTMCell
        :param num_units: int, hidden unit of RNN cell
        :param num_layers: int, the number of layers
        :param inputs: Tensor, the input sequence, shape: [batch_size, max_time_step, num_feature]
        :param seq_lengths: list or 1-D Tensor, sequence length, a list of sequence lengths, the length of the list is batch_size
        :return: the output of last layer bidirectional rnn with concatenating
        """
        _inputs = inputs
        if len(_inputs.get_shape().as_list()) != 3:
            raise ValueError("the inputs must be 3-dimensional Tensor")

        for _ in range(num_layers):
            with tf.variable_scope(None, default_name="bidirectional-rnn"):
                rnn_cell_fw = rnn(num_units)
                rnn_cell_bw = rnn(num_units)
                (output, state) = tf.nn.bidirectional_dynamic_rnn(
                    rnn_cell_fw,
                    rnn_cell_bw,
                    _inputs,
                    seq_lengths,
                    dtype=tf.float32
                )
                _inputs = tf.concat(output, 2)

        return _inputs

    def batch_norm(self, name, x):
        return tf.layers.batch_normalization(x, training=self.mode == RunMode.Trains, fused=True, name=name)

    def conv_block(self, x, growth_rate, dropout_rate=None):
        _x = self.batch_norm(name=None, x=x)
        _x = self.leaky_relu(_x)

        _x = tf.layers.conv2d(
            inputs=_x,
            filters=growth_rate,
            kernel_size=3,
            strides=(1, 1),
            padding='SAME',
            kernel_initializer=self.msra_initializer(3, growth_rate)
        )
        if dropout_rate is not None:
            _x = tf.nn.dropout(_x, dropout_rate)
        return _x

    def dense_block(self, x, nb_layers, growth_rate, nb_filter, dropout_rate=0.2):
        for i in range(nb_layers):
            cb = self.conv_block(x, growth_rate, dropout_rate)
            x = tf.concat([x, cb], 3)
            nb_filter += growth_rate
        return x, nb_filter

    def transition_block(self, x, filters, dropout_kp=None, pool_type=1):
        _x = self.batch_norm(name=None, x=x)
        _x = self.leaky_relu(_x)
        _x = tf.layers.conv2d(
            inputs=_x,
            filters=filters,
            kernel_size=1,
            strides=(1, 1),
            padding='SAME',
            kernel_initializer=self.msra_initializer(3, filters)
        )
        if dropout_kp is not None:
            _x = tf.nn.dropout(_x, dropout_kp)
        if pool_type == 2:
            _x = tf.nn.avg_pool(_x, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")
        elif pool_type == 1:
            _x = tf.nn.avg_pool(_x, [1, 2, 2, 1], [1, 2, 1, 1], "SAME")
        elif pool_type == 3:
            _x = tf.nn.avg_pool(_x, [1, 2, 2, 1], [1, 1, 2, 1], "SAME")
        return _x, filters
