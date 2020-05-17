#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import math
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.regularizers import l2, l1_l2, l1
from config import *


class NetworkUtils(object):
    """
    :param: is_training: 设置为 True 的原因请见 core.py
    """

    def __init__(self, mode: RunMode):
        self.mode = mode
        self.is_training = True

    class BatchNormalization(tf.layers.BatchNormalization):
        """Fixed default name of BatchNormalization"""

        def call(self, *args, **kwargs):
            outputs = super(NetworkUtils.BatchNormalization, self).call(*args, **kwargs)
            # A temporary compatibility with keras batch norm.
            for u in self.updates:
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, u)
            return outputs

    @staticmethod
    def msra_initializer(kl, dl):
        """ MSRA weight initializer
        (https://arxiv.org/pdf/1502.01852.pdf)
        Keyword arguments:
        kl -- kernel size
        dl -- filter numbers
        """

        stddev = math.sqrt(2. / (kl ** 2 * dl))
        return tf.keras.initializers.TruncatedNormal(stddev=stddev)

    def reshape_layer(self, input_tensor, loss_func, shape_list):
        if loss_func == LossFunction.CTC:
            output_tensor = tf.keras.layers.TimeDistributed(
                layer=tf.keras.layers.Flatten(),
            )(inputs=input_tensor, training=self.is_training)
        elif loss_func == LossFunction.CrossEntropy:
            output_tensor = tf.keras.layers.Reshape([shape_list[1], shape_list[2] * shape_list[3]])(input_tensor)
        else:
            raise exception("The current loss function is not supported.", ConfigException.LOSS_FUNC_NOT_SUPPORTED)
        return output_tensor

    def cnn_layer(self, index, inputs, filters, kernel_size, strides):
        """卷积-BN-激活函数-池化结构生成器"""

        with tf.keras.backend.name_scope('unit-{}'.format(index + 1)):
            x = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides[0],
                kernel_regularizer=l1(0.01),
                kernel_initializer=self.msra_initializer(kernel_size, filters),
                padding='same',
                name='cnn-{}'.format(index + 1),
            )(inputs)

            x = self.BatchNormalization(
                fused=True,
                renorm_clipping={
                    'rmax': 3,
                    'rmin': 0.3333,
                    'dmax': 5
                } if index == 0 else None,
                epsilon=1.001e-5,
                name='bn{}'.format(index + 1))(x, training=self.is_training)
            x = tf.keras.layers.LeakyReLU(0.01)(x)
            x = tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2),
                strides=strides[1],
                padding='same',
            )(x)
        return x

    def dense_building_block(self, input_tensor, growth_rate, name, dropout_rate=None):
        """A building block for a dense block.

        # Arguments
            input_tensor: input tensor.
            growth_rate: float, growth rate at dense layers.
            name: string, block label.

        # Returns
            Output tensor for the block.
        """
        # 1x1 Convolution (Bottleneck layer)
        x = self.BatchNormalization(
            epsilon=1.001e-5, name=name + '_0_bn'
        )(input_tensor, training=self.is_training)
        x = tf.keras.layers.LeakyReLU(0.01, name=name + '_0_relu')(x)
        x = tf.keras.layers.Conv2D(
            filters=4 * growth_rate,
            kernel_size=1,
            use_bias=False,
            padding='same',
            name=name + '_1_conv')(x)

        if dropout_rate:
            x = tf.keras.layers.Dropout(dropout_rate)(x)

        # 3x3 Convolution
        x = self.BatchNormalization(
            epsilon=1.001e-5, name=name + '_1_bn'
        )(input_tensor, training=self.is_training)
        x = tf.keras.layers.LeakyReLU(0.01, name=name + '_1_relu')(x)
        x = tf.keras.layers.Conv2D(
            filters=growth_rate,
            kernel_size=3,
            padding='same',
            use_bias=False,
            name=name + '_2_conv')(x)

        if dropout_rate:
            x = tf.keras.layers.Dropout(dropout_rate)(x)

        x = tf.keras.layers.Concatenate(name=name + '_concat')([input_tensor, x])
        return x

    def dense_block(self, input_tensor, blocks, name):
        """A dense block.

        # Arguments
            input_tensor: input tensor.
            blocks: integer, the number of building blocks.
            name: string, block label.

        # Returns conv_block
            output tensor for the block.
        """
        for i in range(blocks):
            input_tensor = self.dense_building_block(input_tensor, 32, name=name + '_block' + str(i + 1))
        return input_tensor

    def transition_block(self, input_tensor, reduction, name):
        """A transition block.

        # Arguments
            input_tensor: input tensor.
            reduction: float, compression rate at transition layers.
            name: string, block label.

        # Returns
            output tensor for the block.
        """
        x = self.BatchNormalization(
            epsilon=1.001e-5, name=name + '_bn'
        )(input_tensor, training=self.is_training)
        x = tf.keras.layers.LeakyReLU(0.01)(x)
        x = tf.keras.layers.Conv2D(
            filters=int(tf.keras.backend.int_shape(x)[3] * reduction),
            kernel_size=1,
            use_bias=False,
            padding='same',
            name=name + '_conv')(x)
        x = tf.keras.layers.AveragePooling2D(2, strides=2, name=name + '_pool', padding='same')(x)
        return x

    def residual_building_block(self, input_tensor, kernel_size, filters, stage, block, strides=(2, 2), s1=True,
                                s2=True):
        """A block that has a conv layer at shortcut.

        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of
                middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
            strides: Strides for the first conv layer in the block.

        # Returns
            Output tensor for the block.

        Note that from stage 3,
        the first conv layer at main path is with strides=(2, 2)
        And the shortcut should have strides=(2, 2) as well
        """
        filters1, filters2, filters3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = tf.keras.layers.Conv2D(
            filters=filters1,
            kernel_size=(1, 1),
            strides=strides,
            kernel_initializer='he_normal',
            padding='same',
            name=conv_name_base + '2a')(input_tensor)
        x = self.BatchNormalization(name=bn_name_base + '2a')(x, training=self.is_training)
        x = tf.keras.layers.LeakyReLU(0.01)(x)

        x = tf.keras.layers.Conv2D(
            filters=filters2,
            kernel_size=kernel_size,
            padding='same',
            kernel_initializer='he_normal',
            name=conv_name_base + '2b')(x)
        x = self.BatchNormalization(name=bn_name_base + '2b')(x, training=self.is_training)
        x = tf.keras.layers.LeakyReLU(0.01)(x)

        x = tf.keras.layers.Conv2D(
            filters=filters3,
            kernel_size=(1, 1),
            kernel_initializer='he_normal',
            padding='same',
            name=conv_name_base + '2c')(x)
        x = self.BatchNormalization(name=bn_name_base + '2c')(x, training=self.is_training)
        shortcut = tf.keras.layers.Conv2D(
            filters=filters3,
            kernel_size=(1, 1),
            strides=strides,
            kernel_initializer='he_normal',
            padding='same',
            name=conv_name_base + '1')(input_tensor)
        shortcut = self.BatchNormalization(
            name=bn_name_base + '1'
        )(shortcut, training=self.is_training)

        x = tf.keras.layers.add([x, shortcut])
        x = tf.keras.layers.LeakyReLU(0.01)(x)
        return x

    def identity_block(self, input_tensor, kernel_size, filters, stage, block):
        """The identity block is the block that has no conv layer at shortcut.

        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of
                middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names

        # Returns
            Output tensor for the block.
        """
        filters1, filters2, filters3 = filters
        bn_axis = 3
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        x = tf.keras.layers.Conv2D(
            filters=filters1,
            kernel_size=(1, 1),
            kernel_initializer='he_normal',
            padding='same',
            name=conv_name_base + '2a'
        )(input_tensor)
        x = self.BatchNormalization(
            # axis=bn_axis,
            name=bn_name_base + '2a',
        )(x, training=self.is_training)
        x = tf.keras.layers.LeakyReLU(0.01)(x)

        x = tf.keras.layers.Conv2D(
            filters=filters2,
            kernel_size=kernel_size,
            padding='same',
            kernel_initializer='he_normal',
            name=conv_name_base + '2b'
        )(x)
        x = self.BatchNormalization(
            # axis=bn_axis,
            name=bn_name_base + '2b',
        )(x, training=self.is_training)
        x = tf.keras.layers.LeakyReLU(0.01)(x)

        x = tf.keras.layers.Conv2D(
            filters=filters3,
            kernel_size=(1, 1),
            padding='same',
            kernel_initializer='he_normal',
            name=conv_name_base + '2c')(x)
        x = self.BatchNormalization(
            # axis=bn_axis,
            name=bn_name_base + '2c',
        )(x, training=self.is_training)
        x = tf.keras.layers.add([x, input_tensor])
        x = tf.keras.layers.LeakyReLU(0.01)(x)
        return x

    @staticmethod
    def _make_divisible(v, divisor, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def inverted_res_block(self, input_tensor, expansion, stride, filters, block_id):
        channel_axis = 1 if tf.keras.backend.image_data_format() == 'channels_first' else -1

        in_channels = tf.keras.backend.int_shape(input_tensor)[channel_axis]
        pointwise_filters = int(filters)
        x = input_tensor
        prefix = 'block_{}_'.format(block_id)

        if block_id:
            # Expand
            x = tf.keras.layers.Conv2D(
                expansion * in_channels,
                kernel_size=1,
                padding='same',
                use_bias=False,
                activation=None,
                name=prefix + 'expand'
            )(x)
            x = self.BatchNormalization(
                epsilon=1e-3,
                momentum=0.999,
                name=prefix + 'expand_BN',
            )(x, training=self.is_training)
            x = tf.keras.layers.LeakyReLU(0.01)(x)
        else:
            prefix = 'expanded_conv_'

        # Depthwise
        x = tf.keras.layers.DepthwiseConv2D(
            kernel_size=3,
            strides=stride,
            activation=None,
            use_bias=False,
            padding='same',
            name=prefix + 'depthwise'
        )(x)
        x = self.BatchNormalization(
            epsilon=1e-3,
            momentum=0.999,
            name=prefix + 'expand_BN',
        )(x, training=self.is_training)

        x = tf.keras.layers.LeakyReLU(0.01)(x)

        # Project
        x = tf.keras.layers.Conv2D(
            pointwise_filters,
            kernel_size=1,
            padding='same',
            use_bias=False,
            activation=None,
            name=prefix + 'project'
        )(x)
        x = self.BatchNormalization(
            epsilon=1e-3,
            momentum=0.999,
            name=prefix + 'expand_BN',
        )(x, training=self.is_training)

        if in_channels == pointwise_filters and stride == 1:
            return tf.keras.layers.Add(name=prefix + 'add')([input_tensor, x])
        return x
