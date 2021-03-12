#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import sys
from config import RecurrentNetwork, RESIZE_MAP, CNNNetwork, Optimizer
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
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()


class NeuralNetwork(object):

    """
    神经网络构建类
    """
    def __init__(self, model_conf: ModelConfig, mode: RunMode, backbone: CNNNetwork, recurrent: RecurrentNetwork):
        """

        :param model_conf: 模型配置
        :param mode: 运行模式 (Trains/Validation/Predict)
        :param backbone:
        :param recurrent:
        """
        self.model_conf = model_conf
        self.decoder = Decoder(self.model_conf)
        self.mode = mode
        self.network = backbone
        self.recurrent = recurrent
        self.inputs = tf.keras.Input(dtype=tf.float32, shape=self.input_shape, name='input')
        self.labels = tf.keras.Input(dtype=tf.int32, shape=[None], sparse=True, name='labels')
        self.utils = NetworkUtils(mode)
        self.merged_summary = None
        self.optimizer = None
        self.dataset_size = None

    @property
    def input_shape(self):
        """
        :return: tuple/list 类型，输入的 Shape
        """
        return RESIZE_MAP[self.model_conf.loss_func](*self.model_conf.resize) + [self.model_conf.image_channel]

    def build_graph(self):
        """
        在当前Session中构建网络计算图
        """
        self._build_model()

    def build_train_op(self, dataset_size=None):
        self.dataset_size = dataset_size
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

        self.seq_len = tf.compat.v1.fill([tf.shape(x)[0]], tf.shape(x)[1], name="seq_len")

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
                self.outputs = FullConnectedRNN(model_conf=self.model_conf, outputs=logits).build()
            elif self.model_conf.loss_func == LossFunction.CrossEntropy:
                self.outputs = FullConnectedCNN(model_conf=self.model_conf, outputs=logits).build()
        return self.outputs

    @property
    def decay_steps(self):
        if not self.dataset_size:
            return 10000
        # return 10000
        epoch_step = int(self.dataset_size / self.model_conf.batch_size)
        return int(epoch_step / 4)

    def _build_train_op(self):
        """构建训练操作符"""

        # 步数
        self.global_step = tf.compat.v1.train.get_or_create_global_step()

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

        # 学习率 指数衰减法
        self.lrn_rate = tf.compat.v1.train.exponential_decay(
            self.model_conf.trains_learning_rate,
            self.global_step,
            staircase=True,
            decay_steps=self.decay_steps,
            decay_rate=0.98,
        )
        tf.compat.v1.summary.scalar('learning_rate', self.lrn_rate)

        if self.model_conf.neu_optimizer == Optimizer.AdaBound:
            self.optimizer = AdaBoundOptimizer(
                learning_rate=self.lrn_rate,
                final_lr=0.001,
                beta1=0.9,
                beta2=0.999,
                amsbound=True
            )
        elif self.model_conf.neu_optimizer == Optimizer.Adam:
            self.optimizer = tf.compat.v1.train.AdamOptimizer(
                learning_rate=self.lrn_rate
            )
        elif self.model_conf.neu_optimizer == Optimizer.RAdam:
            self.optimizer = RAdamOptimizer(
                learning_rate=self.lrn_rate,
                warmup_proportion=0.1,
                min_lr=1e-6
            )
        elif self.model_conf.neu_optimizer == Optimizer.Momentum:
            self.optimizer = tf.compat.v1.train.MomentumOptimizer(
                learning_rate=self.lrn_rate,
                use_nesterov=True,
                momentum=0.9,
            )
        elif self.model_conf.neu_optimizer == Optimizer.SGD:
            self.optimizer = tf.compat.v1.train.GradientDescentOptimizer(
                learning_rate=self.lrn_rate,
            )
        elif self.model_conf.neu_optimizer == Optimizer.AdaGrad:
            self.optimizer = tf.compat.v1.train.AdagradOptimizer(
                learning_rate=self.lrn_rate,
            )
        elif self.model_conf.neu_optimizer == Optimizer.RMSProp:
            self.optimizer = tf.compat.v1.train.RMSPropOptimizer(
                learning_rate=self.lrn_rate,
            )

        # BN 操作符更新(moving_mean, moving_variance)
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)

        # 将 train_op 和 update_ops 融合
        with tf.control_dependencies(update_ops):
            self.train_op = self.optimizer.minimize(
                    loss=self.cost,
                    global_step=self.global_step,
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
    pass
