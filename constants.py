#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
from enum import Enum, unique


@unique
class LabelFrom(Enum):
    """标签来源枚举"""
    XML = 'XML'
    LMDB = 'LMDB'
    FileName = 'FileName'


@unique
class LossFunction(Enum):
    """损失函数枚举"""
    CTC = 'CTC'
    CrossEntropy = 'CrossEntropy'


@unique
class ModelScene(Enum):
    """模型场景枚举"""
    Classification = 'Classification'


@unique
class ModelField(Enum):
    """模型类别枚举"""
    Image = 'Image'
    Text = 'Text'


@unique
class RunMode(Enum):
    """运行模式枚举"""
    Validation = 'Validation'
    Trains = 'Trains'
    Predict = 'Predict'


@unique
class CNNNetwork(Enum):
    """卷积层枚举"""
    CNNX = 'CNNX'
    CNN5 = 'CNN5'
    CNNm6 = 'CNNm6'
    CNNm4 = 'CNNm4'
    ResNet = 'ResNet'
    DenseNet = 'DenseNet'


@unique
class RecurrentNetwork(Enum):
    """循环层枚举"""
    LSTM = 'LSTM'
    BiLSTM = 'BiLSTM'
    GRU = 'GRU'
    BiGRU = 'BiGRU'
    LSTMcuDNN = 'LSTMcuDNN'
    BiLSTMcuDNN = 'BiLSTMcuDNN'
    GRUcuDNN = 'GRUcuDNN'
    NoRecurrent = 'null'


@unique
class Optimizer(Enum):
    """优化器枚举"""
    AdaBound = 'AdaBound'
    Adam = 'Adam'
    Momentum = 'Momentum'
    SGD = 'SGD'
    AdaGrad = 'AdaGrad'
    RMSProp = 'RMSProp'


@unique
class SimpleCharset(Enum):
    """简单字符分类枚举"""
    NUMERIC = 'NUMERIC'
    ALPHANUMERIC = 'ALPHANUMERIC'
    ALPHANUMERIC_LOWER = 'ALPHANUMERIC_LOWER'
    ALPHANUMERIC_UPPER = 'ALPHANUMERIC_UPPER'
    ALPHABET_LOWER = 'ALPHABET_LOWER'
    ALPHABET_UPPER = 'ALPHABET_UPPER'
    ALPHABET = 'ALPHABET'
    ARITHMETIC = 'ARITHMETIC'
    ALPHANUMERIC_LOWER_MIX_ARITHMETIC = 'ALPHANUMERIC_LOWER_MIX_ARITHMETIC'
    FLOAT = 'FLOAT'
    CHINESE_3500 = 'CHINESE_3500'
    ALPHANUMERIC_LOWER_MIX_CHINESE_3500 = 'ALPHANUMERIC_LOWER_MIX_CHINESE_3500'

