#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import tensorflow as tf
from network.utils import NetworkUtils
from config import IMAGE_CHANNEL
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


class CNNm6(object):

    def __init__(self, inputs: tf.Tensor, utils: NetworkUtils):
        self.inputs = inputs
        self.utils = utils
        self.filters = [32, 64, 128, 128, 64]
        self.conv_strides = [1, 1, 1, 1, 1]
        self.pool_strides = {
            4: [(1, 1), (2, 2), (1, 1), (2, 2), (1, 2)],
            6: [(1, 1), (2, 2), (1, 1), (3, 2), (1, 2)],
            8: [(1, 1), (2, 2), (1, 1), (2, 2), (2, 2)],
            10: [(1, 1), (2, 2), (1, 1), (5, 2), (1, 2)],
            12: [(1, 1), (2, 2), (1, 1), (2, 2), (3, 2)],
            16: [(1, 1), (2, 2), (2, 1), (2, 2), (2, 2)],
            18: [(1, 1), (2, 2), (1, 1), (3, 2), (3, 2)],
            24: [(1, 1), (2, 2), (2, 1), (3, 2), (2, 2)],
        }
        self.kernel_size = [7, 5, 3, 3, 3]
        self.trainable = True
        self.renorm = [False, False, True, False, False]

    def block(self, w, inputs, filters, kernel_size, conv_strides, clipping=False, re=True, index=0):

        with tf.variable_scope('unit-{}'.format(index + 1)):
            x = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=conv_strides,
                kernel_regularizer=l2(0.01),
                activity_regularizer=l2(0.01),
                kernel_initializer=self.utils.msra_initializer(kernel_size, filters if index != 0 else 1),
                padding='SAME',
                name='cnn-{}'.format(index + 1),
            )(inputs)
            bn = tf.layers.BatchNormalization(
                renorm=re,
                fused=True,
                renorm_clipping={
                    'rmax': 3,
                    'rmin': 0.3333,
                    'dmax': 5
                } if index == 0 else None,
                epsilon=1.001e-5,
                trainable=self.trainable,
                name='bn{}'.format(index + 1)
            )
            x = bn(x, training=self.utils.training)

            # Implementation of Keras
            # bn = tf.keras.layers.BatchNormalization(
            #     renorm=re,
            #     fused=True,
            #     renorm_clipping={
            #         'rmax': 3,
            #         'rmin': 0.3333,
            #         'dmax': 5
            #     } if clipping else None,
            #     epsilon=1.001e-5,
            #     trainable=self.trainable,
            #     name='bn{}'.format(index + 1)
            # )
            # x = bn(x, training=self.utils.training)
            # for op in bn.updates:
            #     tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, op)

            def logical_sec(a=None, b=None):
                if not b:
                    return tf.less(w, tf.constant(a))
                if not a:
                    return tf.greater_equal(w, tf.constant(b))
                return tf.logical_and(tf.greater_equal(w, tf.constant(a)), tf.less(w, tf.constant(b)))

            x = tf.keras.layers.LeakyReLU(0.01)(x)
            x = tf.case(
                {
                    logical_sec(a=60): lambda: self.max_pooling(x, 4, index),
                    logical_sec(a=60, b=90): lambda: self.max_pooling(x, 6, index),
                    logical_sec(a=90, b=120): lambda: self.max_pooling(x, 8, index),
                    logical_sec(a=120, b=150): lambda: self.max_pooling(x, 10, index),
                    logical_sec(a=150, b=180): lambda: self.max_pooling(x, 12, index),
                    logical_sec(a=180, b=240): lambda: self.max_pooling(x, 16, index),
                    logical_sec(a=240, b=300): lambda: self.max_pooling(x, 18, index),
                    logical_sec(b=300): lambda: self.max_pooling(x, 24, index),
                },
                exclusive=True
            )
        return x

    def max_pooling(self, x, section: int, index):
        x = tf.keras.layers.MaxPooling2D(
            padding='SAME',
            pool_size=(2, 2),
            strides=self.pool_strides[section][index]
        )(x)
        return x

    def build(self):
        with tf.compat.v1.variable_scope('CNNm6'):
            x = self.inputs
            w = tf.shape(self.inputs)[1]

            for i in range(5):
                x = self.block(
                    w=w,
                    inputs=x,
                    filters=self.filters[i],
                    kernel_size=self.kernel_size[i],
                    conv_strides=self.conv_strides[i],
                    re=self.renorm[i],
                    index=i
                )

            shape_list = x.get_shape().as_list()
            print("x.get_shape()", shape_list)
            # tf.multiply(tf.shape(x)[2], shape_list[3])
            x = tf.reshape(x, [tf.shape(x)[0], -1, tf.multiply(shape_list[2], shape_list[3])])
            return x


class CNNm4(object):

    def __init__(self, inputs: tf.Tensor, utils: NetworkUtils):
        self.inputs = inputs
        self.utils = utils
        self.filters = [32, 64, 128, 128, 64]
        self.conv_strides = [1, 1, 1, 1, 1]
        self.pool_strides = {
            6: [(1, 1), (2, 2), (1, 1), (3, 2), (1, 2)],
            8: [(1, 1), (2, 2), (1, 1), (2, 2), (2, 2)],
            12: [(1, 1), (2, 2), (1, 1), (2, 2), (3, 2)],
            16: [(1, 1), (2, 2), (2, 1), (2, 2), (2, 2)],
            18: [(1, 1), (2, 2), (1, 1), (3, 2), (3, 2)],
            24: [(1, 1), (2, 2), (2, 1), (3, 2), (2, 2)],
            32: [(1, 1), (2, 2), (2, 1), (2, 2), (4, 2)],
            36: [(1, 1), (2, 2), (2, 1), (3, 2), (3, 2)],
        }
        self.kernel_size = [7, 5, 3, 3, 3]

    def block(self, w, inputs, filters, kernel_size, conv_strides, index=0):

        with tf.variable_scope('unit-{}'.format(index + 1)):
            x = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=(kernel_size, kernel_size-2),
                strides=conv_strides,
                kernel_regularizer=l2(0.01),
                # activity_regularizer=l2(0.01),
                kernel_initializer=self.utils.msra_initializer(kernel_size, filters if index != 0 else 1),
                padding='SAME',
                name='cnn-{}'.format(index + 1),
            )(inputs)
            bn = tf.layers.BatchNormalization(
                renorm=True if index == 0 else False,
                fused=True,
                renorm_clipping={
                    'rmax': 3,
                    'rmin': 0.3333,
                    'dmax': 5
                } if index == 0 else None,
                epsilon=1.001e-5,
                name='bn{}'.format(index + 1)
            )
            x = bn(x, training=self.utils.training)

            # Implementation of Keras
            # bn = tf.keras.layers.BatchNormalization(
            #     renorm=re,
            #     fused=True,
            #     renorm_clipping={
            #         'rmax': 3,
            #         'rmin': 0.3333,
            #         'dmax': 5
            #     } if clipping else None,
            #     epsilon=1.001e-5,
            #     trainable=self.trainable,
            #     name='bn{}'.format(index + 1)
            # )
            # x = bn(x, training=self.utils.training)
            # for op in bn.updates:
            #     tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, op)

            def logical_sec(a=None, b=None):
                if not b:
                    return tf.less(w, tf.constant(a))
                if not a:
                    return tf.greater_equal(w, tf.constant(b))
                return tf.logical_and(tf.greater_equal(w, tf.constant(a)), tf.less(w, tf.constant(b)))

            x = tf.keras.layers.LeakyReLU(0.01)(x)

            x = tf.case(
                {
                    logical_sec(a=60): lambda: self.max_pooling(x, 6, index),
                    logical_sec(a=60, b=90): lambda: self.max_pooling(x, 8, index),
                    logical_sec(a=90, b=130): lambda: self.max_pooling(x, 12, index),
                    logical_sec(a=130, b=140): lambda: self.max_pooling(x, 16, index),
                    logical_sec(a=140, b=190): lambda: self.max_pooling(x, 18, index),
                    logical_sec(a=190, b=260): lambda: self.max_pooling(x, 24, index),
                    logical_sec(a=260, b=300): lambda: self.max_pooling(x, 32, index),
                    logical_sec(b=300): lambda: self.max_pooling(x, 36, index),
                },
                exclusive=True
            )
        return x

    def max_pooling(self, x, section: int, index):
        x = tf.keras.layers.MaxPooling2D(
            padding='SAME',
            pool_size=(2, 2),
            strides=self.pool_strides[section][index]
        )(x)
        return x

    def build(self):
        with tf.compat.v1.variable_scope('CNNm4'):
            x = self.inputs
            w = tf.shape(self.inputs)[1]
            for i in range(5):
                x = self.block(
                    w=w,
                    inputs=x,
                    filters=self.filters[i],
                    kernel_size=self.kernel_size[i],
                    conv_strides=self.conv_strides[i],
                    index=i
                )

            shape_list = x.get_shape().as_list()
            print("x.get_shape()", shape_list)
            # tf.multiply(tf.shape(x)[2], shape_list[3])
            x = tf.reshape(x, [tf.shape(x)[0], -1, tf.multiply(shape_list[2], shape_list[3])])
            return x