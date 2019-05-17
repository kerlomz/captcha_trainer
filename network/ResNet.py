#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>

import tensorflow as tf
from network.utils import NetworkUtils


class ResNet50(object):

    def __init__(self, inputs: tf.Tensor, utils: NetworkUtils):
        self.inputs = inputs
        self.utils = utils

    def build(self):
        x = self.utils.zero_padding(self.inputs, (3, 3))

        # Stage 1
        _, _, _, in_channels = x.shape.as_list()

        a1 = self.utils.conv2d(
            x=x,
            name="conv1",
            filter_size=7,
            in_channels=in_channels,
            out_channels=64,
            strides=2,
            padding='VALID'
        )

        a1 = self.utils.batch_norm(x=a1, name='bn_conv1')
        # a1 = tf.nn.relu(a1)
        a1 = self.utils.leaky_relu(a1)

        a1 = tf.nn.max_pool(a1, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='VALID')

        # Stage 2
        a2 = self.utils.convolutional_block(a1, f=3, out_channels=[64, 64, 256], stage=2, block='a', s=1)
        a2 = self.utils.identity_block(a2, f=3, out_channels=[64, 64, 256], stage=2, block='b')
        a2 = self.utils.identity_block(a2, f=3, out_channels=[64, 64, 256], stage=2, block='c')

        # Stage 3
        a3 = self.utils.convolutional_block(a2, 3, [128, 128, 512], stage=3, block='a', s=2)
        a3 = self.utils.identity_block(a3, 3, [128, 128, 512], stage=3, block='b')
        a3 = self.utils.identity_block(a3, 3, [128, 128, 512], stage=3, block='c')
        a3 = self.utils.identity_block(a3, 3, [128, 128, 512], stage=3, block='d')

        # Stage 4
        a4 = self.utils.convolutional_block(a3, 3, [256, 256, 1024], stage=4, block='a', s=2)
        a4 = self.utils.identity_block(a4, 3, [256, 256, 1024], stage=4, block='b')
        a4 = self.utils.identity_block(a4, 3, [256, 256, 1024], stage=4, block='c')
        a4 = self.utils.identity_block(a4, 3, [256, 256, 1024], stage=4, block='d')
        a4 = self.utils.identity_block(a4, 3, [256, 256, 1024], stage=4, block='e')
        a4 = self.utils.identity_block(a4, 3, [256, 256, 1024], stage=4, block='f')

        # Stage 5
        a5 = self.utils.convolutional_block(a4, 3, [512, 512, 2048], stage=5, block='a', s=2)
        a5 = self.utils.identity_block(a5, 3, [512, 512, 2048], stage=5, block='b')
        x = self.utils.identity_block(a5, 3, [512, 512, 2048], stage=5, block='c')

        shape_list = x.get_shape().as_list()
        x = tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1] * shape_list[2], shape_list[3]])
        return x
