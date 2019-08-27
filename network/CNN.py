#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import tensorflow as tf
from network.utils import NetworkUtils
from config import IMAGE_CHANNEL
from tensorflow.python.keras.layers.pooling import MaxPooling2D, AveragePooling2D
from tensorflow.python.keras.regularizers import l1, l2, l1_l2


class CNN5(object):

    def __init__(self, inputs: tf.Tensor, utils: NetworkUtils):
        self.inputs = inputs
        self.utils = utils
        # (in_channels, out_channels)
        self.filters = [(IMAGE_CHANNEL, 32), (32, 64), (64, 128), (128, 128), (128, 64)]
        # (conv2d_strides, max_pool_strides)
        self.strides = [(1, 1), (1, 2), (1, 2), (1, 2), (1, 2)]
        self.kernel_size = [7, 5, 3, 3, 3]

    def build(self):
        with tf.compat.v1.variable_scope("CNN5"):
            x = self.inputs

            x = self.utils.cnn_layers(
                inputs=x,
                kernel_size=self.kernel_size,
                filters=self.filters,
                strides=self.strides,
                training=self.utils.training
            )

            shape_list = x.get_shape().as_list()
            print("x.get_shape()", shape_list)
            x = tf.reshape(x, [tf.shape(x)[0], -1, shape_list[2] * shape_list[3]])
            return x


class CNNX(object):

    def __init__(self, inputs: tf.Tensor, utils: NetworkUtils):
        self.inputs = inputs
        self.utils = utils
        self.filters = [32, 64, 128, 128, 64]
        self.strides = [(1, 1), (1, 2), (1, 2), (1, 2), (1, 2)]
        self.kernel_size = [7, 5, 3, 3, 3]
        self.trainable = True

    def block(self, inputs, filters, kernel_size, strides, pooling, pool_size=None, clipping=False, re=True, index=0):
        with tf.variable_scope('unit-{}'.format(index + 1)):
            x = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides[0],
                kernel_regularizer=l2(0.01),
                kernel_initializer=self.utils.msra_initializer(kernel_size, filters if index != 0 else 1),
                padding='SAME',
                name='cnn-{}'.format(index + 1),
            )(inputs)
            x = tf.layers.BatchNormalization(
                renorm=re,
                fused=True,
                renorm_clipping={
                    'rmax': 3,
                    'rmin': 0.3333,
                    'dmax': 5
                } if clipping else None,
                epsilon=1.001e-5,
                trainable=self.trainable,
                name='bn{}'.format(index + 1)
            )(x, training=self.utils.training)

            x = tf.keras.layers.LeakyReLU(0.01)(x)
            if pooling:
                x = pooling(
                    pool_size=pool_size,
                    strides=strides[1],
                    padding='SAME',
                )(x)

        return x

    @staticmethod
    def max_pooling(x, strides=(1, 1), pool_size=(2, 2)):
        x = tf.keras.layers.MaxPooling2D(
            padding='SAME',
            pool_size=pool_size,
            strides=strides
        )(x)
        return x

    @staticmethod
    def avg_pooling(x, strides=(1, 1), pool_size=(2, 2)):
        x = tf.keras.layers.AveragePooling2D(
            padding='SAME',
            pool_size=pool_size,
            strides=strides
        )(x)
        return x

    def build(self):
        with tf.compat.v1.variable_scope('CNNX'):
            x = self.inputs
            w = tf.shape(self.inputs)[1]

            # ====== Layer 1 ======
            x = self.block(
                inputs=x,
                filters=self.filters[0],
                kernel_size=self.kernel_size[0],
                strides=self.strides[0],
                pooling=MaxPooling2D,
                pool_size=(2, 2),
                re=True,
                index=0
            )

            # ====== Layer 2 ======
            x = self.block(
                inputs=x,
                filters=self.filters[1],
                kernel_size=self.kernel_size[1],
                strides=self.strides[1],
                pooling=None,
                re=False,
                index=1
            )
            with tf.variable_scope('unit-{}'.format(2)):
                x = tf.cond(
                    tf.greater_equal(w, tf.constant(150)),
                    lambda: self.max_pooling(x, strides=(2, 2)),
                    lambda: self.max_pooling(x, strides=(1, 2)),
                )

            # ====== Layer 3 ======
            x = self.block(
                inputs=x,
                filters=self.filters[2],
                kernel_size=self.kernel_size[2],
                strides=self.strides[2],
                pooling=None,
                re=False,
                index=2
            )
            with tf.variable_scope('unit-{}'.format(3)):
                x = tf.cond(
                    tf.greater_equal(w, tf.constant(60)),
                    lambda: self.max_pooling(x, strides=(2, 2)),
                    lambda: self.max_pooling(x, strides=(1, 2)),
                )

            # ====== Layer 4 ======
            x = self.block(
                inputs=x,
                filters=self.filters[3],
                kernel_size=self.kernel_size[3],
                strides=self.strides[3],
                pooling=MaxPooling2D,
                pool_size=(2, 2),
                re=False,
                index=3
            )

            # ====== Layer 5 ======
            x = self.block(
                inputs=x,
                filters=self.filters[4],
                kernel_size=self.kernel_size[4],
                strides=self.strides[4],
                pooling=None,
                re=False,
                index=4
            )
            with tf.variable_scope('unit-{}'.format(5)):
                x = tf.cond(
                    tf.greater_equal(tf.shape(self.inputs)[1], tf.constant(180)),
                    lambda: self.max_pooling(x, strides=(3, 2)),
                    lambda: self.max_pooling(x, strides=(2, 2)),
                )

            shape_list = x.get_shape().as_list()
            print("x.get_shape()", shape_list)
            x = tf.reshape(x, [tf.shape(x)[0], -1, shape_list[2] * shape_list[3]])
            return x
