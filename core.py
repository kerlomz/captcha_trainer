#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import sys
from config import *
from network.CNN import *
from network.MobileNet import MobileNetV2
from network.DenseNet import DenseNet
from network.GRU import GRU, BiGRU, GRUcuDNN
from network.LSTM import LSTM, BiLSTM, BiLSTMcuDNN, LSTMcuDNN
from network.ResNet import ResNet50, ResNetTiny
from network.utils import NetworkUtils
from optimizer.AdaBound import AdaBoundOptimizer
from optimizer.RAdam import RAdamOptimizer
from loss import *
from encoder import *
from decoder import *
from fc import *

import tensorflow as tf

weight_decay = 1e-4


def relu(x, name='relu6'):
    return tf.nn.relu6(x, name)


def batch_norm(x, momentum=0.9, epsilon=1e-5, train=True, name='bn'):
    return tf.layers.batch_normalization(x,
                                         momentum=momentum,
                                         epsilon=epsilon,
                                         scale=True,
                                         training=train,
                                         name=name)


def conv2d(input_, output_dim, k_h, k_w, d_h, d_w, stddev=0.02, name='conv2d', bias=False):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        if bias:
            biases = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)

        return conv


def conv2d_block(input, out_dim, k, s, is_train, name):
    with tf.name_scope(name), tf.variable_scope(name):
        net = conv2d(input, out_dim, k, k, s, s, name='conv2d')
        net = batch_norm(net, train=is_train, name='bn')
        net = relu(net)
        return net


def conv_1x1(input, output_dim, name, bias=False):
    with tf.name_scope(name):
        return conv2d(input, output_dim, 1, 1, 1, 1, stddev=0.02, name=name, bias=bias)


def pwise_block(input, output_dim, is_train, name, bias=False):
    with tf.name_scope(name), tf.variable_scope(name):
        out = conv_1x1(input, output_dim, bias=bias, name='pwb')
        out = batch_norm(out, train=is_train, name='bn')
        out = relu(out)
        return out


def dwise_conv(input, k_h=3, k_w=3, channel_multiplier=1, strides=[1, 1, 1, 1],
               padding='SAME', stddev=0.02, name='dwise_conv', bias=False):
    with tf.variable_scope(name):
        in_channel = input.get_shape().as_list()[-1]
        w = tf.get_variable('w', [k_h, k_w, in_channel, channel_multiplier],
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.depthwise_conv2d(input, w, strides, padding, rate=None, name=None, data_format=None)
        if bias:
            biases = tf.get_variable('bias', [in_channel * channel_multiplier],
                                     initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)

        return conv


def res_block(input, expansion_ratio, output_dim, stride, is_train, name, bias=False, shortcut=True):
    with tf.name_scope(name), tf.variable_scope(name):
        # pw
        bottleneck_dim = round(expansion_ratio * input.get_shape().as_list()[-1])
        net = conv_1x1(input, bottleneck_dim, name='pw', bias=bias)
        net = batch_norm(net, train=is_train, name='pw_bn')
        net = relu(net)
        # dw
        net = dwise_conv(net, strides=[1, stride, stride, 1], name='dw', bias=bias)
        net = batch_norm(net, train=is_train, name='dw_bn')
        net = relu(net)
        # pw & linear
        net = conv_1x1(net, output_dim, name='pw_linear', bias=bias)
        net = batch_norm(net, train=is_train, name='pw_linear_bn')

        # element wise add, only for stride==1
        if shortcut and stride == 1:
            in_dim = int(input.get_shape().as_list()[-1])
            if in_dim != output_dim:
                ins = conv_1x1(input, output_dim, name='ex_dim')
                net = ins + net
            else:
                net = input + net

        return net


def separable_conv(input, k_size, output_dim, stride, pad='SAME', channel_multiplier=1, name='sep_conv', bias=False):
    with tf.name_scope(name), tf.variable_scope(name):
        in_channel = input.get_shape().as_list()[-1]
        dwise_filter = tf.get_variable('dw', [k_size, k_size, in_channel, channel_multiplier],
                                       regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                       initializer=tf.truncated_normal_initializer(stddev=0.02))

        pwise_filter = tf.get_variable('pw', [1, 1, in_channel * channel_multiplier, output_dim],
                                       regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                       initializer=tf.truncated_normal_initializer(stddev=0.02))
        strides = [1, stride, stride, 1]

        conv = tf.nn.separable_conv2d(input, dwise_filter, pwise_filter, strides, padding=pad, name=name)
        if bias:
            biases = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)
        return conv


def global_avg(x):
    with tf.name_scope('global_avg'):
        net = tf.layers.average_pooling2d(x, x.get_shape()[1:-1], 1)
        return net


def flatten(x):
    # flattened=tf.reshape(input,[x.get_shape().as_list()[0], -1])  # or, tf.layers.flatten(x)
    return tf.contrib.layers.flatten(x)


def pad2d(inputs, pad=(0, 0), mode='CONSTANT'):
    paddings = [[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]]
    net = tf.pad(inputs, paddings, mode=mode)
    return net
def mobilenetv2(inputs, num_classes, is_train=True, reuse=False):
    exp = 6  # expansion ratio
    with tf.variable_scope('mobilenetv2'):
        net = conv2d_block(inputs, 32, 3, 2, is_train, name='conv1_1')  # size/2

        net = res_block(net, 1, 16, 1, is_train, name='res2_1')

        net = res_block(net, exp, 24, 2, is_train, name='res3_1')  # size/4
        net = res_block(net, exp, 24, 1, is_train, name='res3_2')

        net = res_block(net, exp, 32, 2, is_train, name='res4_1')  # size/8
        net = res_block(net, exp, 32, 1, is_train, name='res4_2')
        net = res_block(net, exp, 32, 1, is_train, name='res4_3')

        net = res_block(net, exp, 64, 2, is_train, name='res5_1')
        net = res_block(net, exp, 64, 1, is_train, name='res5_2')
        net = res_block(net, exp, 64, 1, is_train, name='res5_3')
        net = res_block(net, exp, 64, 1, is_train, name='res5_4')

        net = res_block(net, exp, 96, 1, is_train, name='res6_1')  # size/16
        net = res_block(net, exp, 96, 1, is_train, name='res6_2')
        net = res_block(net, exp, 96, 1, is_train, name='res6_3')

        net = res_block(net, exp, 160, 2, is_train, name='res7_1')  # size/32
        net = res_block(net, exp, 160, 1, is_train, name='res7_2')
        net = res_block(net, exp, 160, 1, is_train, name='res7_3')

        net = res_block(net, exp, 320, 1, is_train, name='res8_1', shortcut=False)

        net = pwise_block(net, 1280, is_train, name='conv9_1')
        return net
class NeuralNetwork(object):
    """
    神经网络构建类
    """
    def __init__(self, model_conf: ModelConfig, mode: RunMode, cnn: CNNNetwork, recurrent: RecurrentNetwork):
        self.model_conf = model_conf
        self.mode = mode
        self.decoder = Decoder(self.model_conf)
        self.utils = NetworkUtils(mode)
        self.network = cnn
        self.recurrent = recurrent
        print(self.input_shape)
        self.inputs = tf.keras.Input(dtype=tf.float32, shape=self.input_shape, name='input')
        self.labels = tf.keras.Input(dtype=tf.int32, shape=[None], sparse=True, name='labels')
        self.merged_summary = None

    @property
    def input_shape(self):
        """
        :return: tuple/list 类型，输入的Shape
        """
        return RESIZE_MAP[self.model_conf.loss_func](*self.model_conf.resize) + [self.model_conf.image_channel]

    def build_graph(self):
        """在当前Session中构建网络计算图，无返回"""
        self._build_model()
        self._build_train_op()
        self.merged_summary = tf.compat.v1.summary.merge_all()

    def _build_model(self):

        """选择采用哪种卷积网络"""
        if self.network == CNNNetwork.CNN5:
            x = CNN5(model_conf=self.model_conf, inputs=self.inputs, utils=self.utils).build()

        elif self.network == CNNNetwork.CNNX:
            x = CNNX(model_conf=self.model_conf, inputs=self.inputs, utils=self.utils).build()

        elif self.network == CNNNetwork.ResNetTiny:
            x = ResNetTiny(model_conf=self.model_conf, inputs=self.inputs, utils=self.utils).build()

        elif self.network == CNNNetwork.ResNet50:
            x = ResNet50(model_conf=self.model_conf, inputs=self.inputs, utils=self.utils).build()

        elif self.network == CNNNetwork.DenseNet:
            x = DenseNet(model_conf=self.model_conf, inputs=self.inputs, utils=self.utils).build()

        elif self.network == CNNNetwork.MobileNetV2:
            x = MobileNetV2(model_conf=self.model_conf, inputs=self.inputs, utils=self.utils).build()

        else:
            raise ValueError('This cnn neural network is not supported at this time.')

        """选择采用哪种循环网络"""

        # time_major = True: [max_time_step, batch_size, num_classes]
        tf.compat.v1.logging.info("CNN Output: {}".format(x.get_shape()))

        self.seq_len = tf.fill([tf.shape(x)[0]], tf.shape(x)[1], name="seq_len")
        # self.labels_len = tf.fill([BATCH_SIZE], 12, name="labels_len")
        if self.recurrent == RecurrentNetwork.NoRecurrent:
            self.recurrent_network_builder = None
        elif self.recurrent == RecurrentNetwork.LSTM:
            self.recurrent_network_builder = LSTM(model_conf=self.model_conf, inputs=x, utils=self.utils)
        elif self.recurrent == RecurrentNetwork.BiLSTM:
            self.recurrent_network_builder = BiLSTM(model_conf=self.model_conf, inputs=x, utils=self.utils)
        elif self.recurrent == RecurrentNetwork.GRU:
            self.recurrent_network_builder = GRU(model_conf=self.model_conf, inputs=x, utils=self.utils)
        elif self.recurrent == RecurrentNetwork.BiGRU:
            self.recurrent_network_builder = BiGRU(model_conf=self.model_conf, inputs=x, utils=self.utils)
        elif self.recurrent == RecurrentNetwork.LSTMcuDNN:
            self.recurrent_network_builder = LSTMcuDNN(model_conf=self.model_conf, inputs=x, utils=self.utils)
        elif self.recurrent == RecurrentNetwork.BiLSTMcuDNN:
            self.recurrent_network_builder = BiLSTMcuDNN(model_conf=self.model_conf, inputs=x, utils=self.utils)
        elif self.recurrent == RecurrentNetwork.GRUcuDNN:
            self.recurrent_network_builder = GRUcuDNN(model_conf=self.model_conf, inputs=x, utils=self.utils)
        else:
            raise ValueError('This recurrent neural network is not supported at this time.')

        logits = self.recurrent_network_builder.build() if self.recurrent_network_builder else x
        if self.recurrent_network_builder and self.model_conf.loss_func != LossFunction.CTC:
            raise ValueError('CTC loss must use recurrent neural network.')

        """输出层，根据Loss函数区分"""
        with tf.keras.backend.name_scope('output'):
            if self.model_conf.loss_func == LossFunction.CTC:
                self.outputs = FullConnectedRNN(model_conf=self.model_conf, mode=self.mode, outputs=logits).build()
            elif self.model_conf.loss_func == LossFunction.CrossEntropy:
                self.outputs = FullConnectedCNN(model_conf=self.model_conf, mode=self.mode, outputs=logits).build()
            return self.outputs

    def _build_train_op(self):
        """操作符生成器"""
        # 步数
        self.global_step = tf.train.get_or_create_global_step()
        # Loss函数
        if self.model_conf.loss_func == LossFunction.CTC:
            self.loss = Loss.ctc(
                labels=self.labels,
                logits=self.outputs,
                sequence_length=self.seq_len
            )
        elif self.model_conf.loss_func == LossFunction.CrossEntropy:
            self.loss = Loss.cross_entropy(
                labels=self.labels,
                logits=self.outputs
            )

        self.cost = tf.reduce_mean(self.loss)

        tf.compat.v1.summary.scalar('cost', self.cost)

        # 学习率
        self.lrn_rate = tf.compat.v1.train.exponential_decay(
            self.model_conf.trains_learning_rate,
            self.global_step,
            staircase=True,
            decay_steps=10000,
            decay_rate=0.98,
        )
        tf.compat.v1.summary.scalar('learning_rate', self.lrn_rate)

        # 训练参数更新
        update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)

        # Storing adjusted smoothed mean and smoothed variance operations
        with tf.control_dependencies(update_ops):

            # TODO 这种if-else结构感觉很蠢，优化器选择器
            if self.model_conf.neu_optimizer == Optimizer.AdaBound:
                self.train_op = AdaBoundOptimizer(
                    learning_rate=self.lrn_rate,
                    final_lr=0.001,
                    beta1=0.9,
                    beta2=0.999,
                    amsbound=True
                ).minimize(
                    loss=self.cost,
                    global_step=self.global_step
                )
            elif self.model_conf.neu_optimizer == Optimizer.Adam:
                self.train_op = tf.train.AdamOptimizer(
                    learning_rate=self.lrn_rate
                ).minimize(
                    self.cost,
                    global_step=self.global_step
                )
            elif self.model_conf.neu_optimizer == Optimizer.RAdam:
                self.train_op = RAdamOptimizer(
                    learning_rate=self.lrn_rate,
                    warmup_proportion=0.1,
                    min_lr=1e-6
                ).minimize(
                    self.cost,
                    global_step=self.global_step
                )
            elif self.model_conf.neu_optimizer == Optimizer.Momentum:
                self.train_op = tf.train.MomentumOptimizer(
                    learning_rate=self.lrn_rate,
                    use_nesterov=True,
                    momentum=0.9,
                ).minimize(
                    self.cost,
                    global_step=self.global_step
                )
            elif self.model_conf.neu_optimizer == Optimizer.SGD:
                self.train_op = tf.train.GradientDescentOptimizer(
                    learning_rate=self.lrn_rate,
                ).minimize(
                    self.cost,
                    global_step=self.global_step
                )
            elif self.model_conf.neu_optimizer == Optimizer.AdaGrad:
                self.train_op = tf.train.AdagradOptimizer(
                    learning_rate=self.lrn_rate,
                ).minimize(
                    self.cost,
                    global_step=self.global_step
                )
            elif self.model_conf.neu_optimizer == Optimizer.RMSProp:
                self.train_op = tf.train.RMSPropOptimizer(
                    learning_rate=self.lrn_rate,
                ).minimize(
                    self.cost,
                    global_step=self.global_step
                )

        # 转录层-Loss函数
        if self.model_conf.loss_func == LossFunction.CTC:
            self.dense_decoded = self.decoder.ctc(
                inputs=self.outputs,
                sequence_length=self.seq_len
            )
        elif self.model_conf.loss_func == LossFunction.CrossEntropy:
            self.dense_decoded = self.decoder.cross_entropy(
                inputs=self.outputs
            )


if __name__ == '__main__':
    # GraphOCR(RunMode.Trains, CNNNetwork.CNN5, RecurrentNetwork.GRU).build_graph()
    pass
