#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import math
import tensorflow as tf
from tensorflow.python.keras.regularizers import l2, l1_l2
from config import *


class NetworkUtils(object):

    def __init__(self, mode: RunMode):
        self.extra_train_ops = []
        self.mode: RunMode = mode
        self.training = self.mode == RunMode.Trains

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

    def cnn_layers(self, inputs, filters, kernel_size, strides, training=True):
        x = inputs
        for i in range(len(kernel_size)):
            with tf.variable_scope('unit-{}'.format(i + 1)):
                x = tf.keras.layers.Conv2D(
                    filters=filters[i][1],
                    kernel_size=kernel_size[i],
                    strides=strides[i][0],
                    kernel_regularizer=l2(0.01),
                    kernel_initializer=self.msra_initializer(kernel_size[i], filters[i][0]),
                    padding='same',
                    name='cnn-{}'.format(i + 1),
                )(x)
                batch_normalization = tf.layers.BatchNormalization(
                    fused=True,
                    renorm_clipping={
                        'rmax': 3,
                        'rmin': 0.3333,
                        'dmax': 5
                    },
                    epsilon=1.001e-5,
                    name='bn{}'.format(i + 1)
                )

                x = batch_normalization(x, training=training)
                x = tf.keras.layers.LeakyReLU(0.01)(x)
                x = tf.keras.layers.MaxPooling2D(
                    pool_size=(2, 2),
                    strides=strides[i][1]
                )(x)
                print(x.shape)
        return x

    def conv_block(self, x, growth_rate, name):
        """A building block for a dense block.

        # Arguments
            x: input tensor.
            growth_rate: float, growth rate at dense layers.
            name: string, block label.

        # Returns
            Output tensor for the block.
        """
        bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1
        x1 = tf.layers.BatchNormalization(
            axis=bn_axis,
            epsilon=1.001e-5,
            name=name + '_0_bn'
        )(x, training=self.training)
        x1 = tf.keras.layers.Activation('relu', name=name + '_0_relu')(x1)
        x1 = tf.keras.layers.Conv2D(
            4 * growth_rate, 1,
            kernel_regularizer=l2(0.01),
            bias_regularizer=l2(0.005),
            use_bias=False,
            kernel_initializer=self.msra_initializer(3, growth_rate),
            name=name + '_1_conv'
        )(x1)
        x1 = tf.layers.BatchNormalization(
            axis=bn_axis,
            epsilon=1.001e-5,
            name=name + '_1_bn'
        )(x1, training=self.training)
        x1 = tf.keras.layers.Activation(
            'relu',
            name=name + '_1_relu'
        )(x1)
        x1 = tf.keras.layers.Conv2D(
            growth_rate, 3,
            padding='same',
            use_bias=False,
            name=name + '_2_conv'
        )(x1)
        x = tf.keras.layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
        return x

    def dense_block(self, x, blocks, name):
        """A dense block.

        # Arguments
            x: input tensor.
            blocks: integer, the number of building blocks.
            name: string, block label.

        # Returnsconv_block
            output tensor for the block.
        """
        for i in range(blocks):
            x = self.conv_block(x, 32, name=name + '_block' + str(i + 1))
        return x

    def transition_block(self, x, reduction, name):
        """A transition block.

        # Arguments
            x: input tensor.
            reduction: float, compression rate at transition layers.
            name: string, block label.

        # Returns
            output tensor for the block.
        """
        bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1
        x = tf.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_bn')(x, training=self.training)
        x = tf.keras.layers.Activation('relu', name=name + '_relu')(x)
        x = tf.keras.layers.Conv2D(
            filters=int(tf.keras.backend.int_shape(x)[bn_axis] * reduction),
            kernel_size=1,
            use_bias=False,
            kernel_regularizer=l1_l2(0.01),
            name=name + '_conv'
        )(x)
        x = tf.keras.layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
        return x
