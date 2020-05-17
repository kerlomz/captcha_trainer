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


class NeuralNetwork(object):


    """
    神经网络构建类
    大家需要注意，这里有一个极为不专业的操作，
    :param: mode 参数它没有被真正使用，我可太菜了，整了半天BN的is_training还是被打到了。
    说明：在 mode[Validation] 场景下，training=False 验证集的准确率波动很厉害，在 training=True 下准确率却很稳定上升
    training=True 在mode[Predict] 场景下预测 BatchSize=1 的数据也能得到较为接近训练准确率的成绩。
    按我原本的理解，mode[Validation/Predict] 场景下 training 都应该设为 False。
    之前的版本 mode[Validation] 模式其实也是使用 True 值，我尝试过的方法是定义一个 tf.placeholder 来动态传入 is_training 值
    因为是静态图，处理方法感觉还是不够优雅，其实直接用 tf.keras.Model 挺香的，但是由于 tf2 测试过还不是很稳定暂时不打算引入。
    """
    def __init__(self, model_conf: ModelConfig, mode: RunMode, cnn: CNNNetwork, recurrent: RecurrentNetwork):
        self.model_conf = model_conf
        self.decoder = Decoder(self.model_conf)
        self.mode = mode
        self.network = cnn
        self.recurrent = recurrent
        self.inputs = tf.keras.Input(dtype=tf.float32, shape=self.input_shape, name='input')
        self.labels = tf.keras.Input(dtype=tf.int32, shape=[None], sparse=True, name='labels')
        self.utils = NetworkUtils(mode)
        self.merged_summary = None
        self.optimizer = None

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

    def _build_train_op(self):
        """构建训练操作符"""

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

        # 学习率 指数衰减法
        self.lrn_rate = tf.compat.v1.train.exponential_decay(
            self.model_conf.trains_learning_rate,
            self.global_step,
            staircase=True,
            decay_steps=10000,
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
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.lrn_rate
            )
        elif self.model_conf.neu_optimizer == Optimizer.RAdam:
            self.optimizer = RAdamOptimizer(
                learning_rate=self.lrn_rate,
                warmup_proportion=0.1,
                min_lr=1e-6
            )
        elif self.model_conf.neu_optimizer == Optimizer.Momentum:
            self.optimizer = tf.train.MomentumOptimizer(
                learning_rate=self.lrn_rate,
                use_nesterov=True,
                momentum=0.9,
            )
        elif self.model_conf.neu_optimizer == Optimizer.SGD:
            self.optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=self.lrn_rate,
            )
        elif self.model_conf.neu_optimizer == Optimizer.AdaGrad:
            self.optimizer = tf.train.AdagradOptimizer(
                learning_rate=self.lrn_rate,
            )
        elif self.model_conf.neu_optimizer == Optimizer.RMSProp:
            self.optimizer = tf.train.RMSPropOptimizer(
                learning_rate=self.lrn_rate,
            )

        # BN 操作符更新(moving_mean, moving_variance)
        update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)

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
    # GraphOCR(RunMode.Trains, CNNNetwork.CNN5, RecurrentNetwork.GRU).build_graph()
    pass
