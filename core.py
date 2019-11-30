#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import sys
from config import *
from network.CNN import *
from network.DenseNet import DenseNet
from network.GRU import GRU, BiGRU, GRUcuDNN
from network.LSTM import LSTM, BiLSTM, BiLSTMcuDNN, LSTMcuDNN
from network.ResNet import ResNet50
from network.utils import NetworkUtils
from optimizer.AdaBound import AdaBoundOptimizer
from loss import *
from encoder import *
from decoder import *
from fc import *

# slim = tf.contrib.slim


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

        elif self.network == CNNNetwork.CNNm4:
            x = CNNm4(model_conf=self.model_conf, inputs=self.inputs, utils=self.utils).build()

        elif self.network == CNNNetwork.CNNm6:
            x = CNNm6(model_conf=self.model_conf, inputs=self.inputs, utils=self.utils).build()

        elif self.network == CNNNetwork.ResNet:
            x = ResNet50(model_conf=self.model_conf, inputs=self.inputs, utils=self.utils).build()

        elif self.network == CNNNetwork.DenseNet:
            x = DenseNet(model_conf=self.model_conf, inputs=self.inputs, utils=self.utils).build()

        else:
            tf.logging.error('This cnn neural network is not supported at this time.')
            sys.exit(-1)

        """选择采用哪种循环网络"""
        # TODO 这种if-else结构感觉很蠢，循环层
        # time_major = True: [max_time_step, batch_size, num_classes]
        # time_major = False: [batch_size, max_time_step, num_classes]
        tf.compat.v1.logging.info("CNN Output: {}".format(x.get_shape()))

        # self.seq_len = tf.fill([tf.shape(x)[0]], 12, name="seq_len")
        self.seq_len = tf.fill([tf.shape(x)[0]], tf.shape(x)[1], name="seq_len")
        # self.labels_len = tf.fill([BATCH_SIZE], 12, name="labels_len")
        if not self.recurrent:
            self.recurrent_network_builder = None
        elif self.recurrent == RecurrentNetwork.NoRecurrent:
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
            tf.logging.error('This recurrent neural network is not supported at this time.')
            sys.exit(-1)

        logits = self.recurrent_network_builder.build() if self.recurrent_network_builder else x

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
                    final_lr=0.01,
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
