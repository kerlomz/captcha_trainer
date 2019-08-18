#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>

import tensorflow as tf
from tensorflow.contrib.slim import nets
from network.utils import NetworkUtils, RunMode

slim = tf.contrib.slim


class ResNet50(object):

    def __init__(self, inputs: tf.Tensor, utils: NetworkUtils):
        self.inputs = inputs
        self.utils = utils

    def build(self):

        # TensorFlow Implementation Version
        with slim.arg_scope(nets.resnet_v2.resnet_arg_scope()):
            x, endpoints = nets.resnet_v2.resnet_v2_50(
                inputs=self.inputs,
                num_classes=None,
                is_training=self.utils.mode == RunMode.Trains,
                global_pool=False
            )

        # # Keras Implementation Version
        # with slim.arg_scope(nets.resnet_v2.resnet_arg_scope()):
        #     x = tf.keras.applications.resnet50.ResNet50(
        #         include_top=False,
        #         weights=None,
        #         pooling=None,
        #         input_tensor=tf.keras.Input(
        #             tensor=self.inputs,
        #             shape=self.inputs.get_shape().as_list()
        #         )
        #     )(self.inputs)

        print("x.get_shape()", x.get_shape())
        shape_list = x.get_shape().as_list()
        x = tf.keras.layers.Reshape([tf.shape(x)[1] * shape_list[2], shape_list[3]])(x)
        return x
