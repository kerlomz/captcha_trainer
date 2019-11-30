#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>

import tensorflow as tf
from tensorflow.contrib.slim import nets
from network.utils import NetworkUtils, RunMode
from constants import LossFunction
from config import ModelConfig

slim = tf.contrib.slim


class ResNet50(object):
    """ResNet50网络的实现"""
    def __init__(self, model_conf: ModelConfig, inputs: tf.Tensor, utils: NetworkUtils):
        self.model_conf = model_conf
        self.inputs = inputs
        self.utils = utils
        self.loss_func = self.model_conf.loss_func

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
        #     )(self.inputs, training=self.utils.mode == RunMode.Trains)
        #
        # print("x.get_shape()", x.get_shape())
        shape_list = x.get_shape().as_list()
        print(shape_list)
        if self.loss_func == LossFunction.CTC:
            x = tf.keras.layers.Reshape([tf.shape(x)[1], shape_list[2] * shape_list[3]])(x)
        elif self.loss_func == LossFunction.CrossEntropy:
            x = tf.keras.layers.Reshape([shape_list[1],  shape_list[2] * shape_list[3]])(x)
        print(x.shape)
        return x
