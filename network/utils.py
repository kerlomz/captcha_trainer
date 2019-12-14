#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import math
import tensorflow as tf
from tensorflow.python.keras.regularizers import l2, l1_l2, l1
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
        return tf.keras.initializers.TruncatedNormal(stddev=stddev)

    @staticmethod
    def reshape_layer(input_tensor, loss_func, shape_list):
        if loss_func == LossFunction.CTC:
            output_tensor = tf.keras.layers.Reshape([-1, shape_list[2] * shape_list[3]])(input_tensor)
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
            x = tf.layers.BatchNormalization(
                fused=True,
                renorm_clipping={
                    'rmax': 3,
                    'rmin': 0.3333,
                    'dmax': 5
                } if index == 0 else None,
                epsilon=1.001e-5,
                name='bn{}'.format(index + 1))(x, training=self.training)
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
        x = tf.layers.BatchNormalization(epsilon=1.001e-5, name=name + '_0_bn')(input_tensor, training=self.training)
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
        x = tf.layers.BatchNormalization(epsilon=1.001e-5, name=name + '_1_bn')(x, training=self.training)
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
        x = tf.layers.BatchNormalization(epsilon=1.001e-5, name=name + '_bn')(input_tensor, training=self.training)
        x = tf.keras.layers.LeakyReLU(0.01)(x)
        x = tf.keras.layers.Conv2D(
            filters=int(tf.keras.backend.int_shape(x)[3] * reduction),
            kernel_size=1,
            use_bias=False,
            padding='same',
            name=name + '_conv')(x)
        x = tf.keras.layers.AveragePooling2D(2, strides=2, name=name + '_pool', padding='same')(x)
        return x

    def residual_building_block(self, input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
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
        x = tf.layers.BatchNormalization(name=bn_name_base + '2a')(x, training=self.training)
        x = tf.keras.layers.LeakyReLU(0.01)(x)

        x = tf.keras.layers.Conv2D(
            filters=filters2,
            kernel_size=kernel_size,
            padding='same',
            kernel_initializer='he_normal',
            name=conv_name_base + '2b')(x)
        x = tf.layers.BatchNormalization(name=bn_name_base + '2b')(x, training=self.training)
        x = tf.keras.layers.LeakyReLU(0.01)(x)

        x = tf.keras.layers.Conv2D(
            filters=filters3,
            kernel_size=(1, 1),
            kernel_initializer='he_normal',
            padding='same',
            name=conv_name_base + '2c')(x)
        x = tf.layers.BatchNormalization(name=bn_name_base + '2c')(x, training=self.training)

        shortcut = tf.keras.layers.Conv2D(
            filters=filters3,
            kernel_size=(1, 1),
            strides=strides,
            kernel_initializer='he_normal',
            padding='same',
            name=conv_name_base + '1')(input_tensor)
        shortcut = tf.layers.BatchNormalization(name=bn_name_base + '1')(shortcut, training=self.training)

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
        x = tf.layers.BatchNormalization(
            axis=bn_axis,
            name=bn_name_base + '2a'
        )(x, training=self.training)
        x = tf.keras.layers.LeakyReLU(0.01)(x)

        x = tf.keras.layers.Conv2D(
            filters=filters2,
            kernel_size=kernel_size,
            padding='same',
            kernel_initializer='he_normal',
            name=conv_name_base + '2b'
        )(x)
        x = tf.layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '2b'
        )(x, training=self.training)
        x = tf.keras.layers.LeakyReLU(0.01)(x)

        x = tf.keras.layers.Conv2D(
            filters=filters3,
            kernel_size=(1, 1),
            padding='same',
            kernel_initializer='he_normal',
            name=conv_name_base + '2c')(x)
        x = tf.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x, training=self.training)
        x = tf.keras.layers.add([x, input_tensor])
        x = tf.keras.layers.LeakyReLU(0.01)(x)
        return x
