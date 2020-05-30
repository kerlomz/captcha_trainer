#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
from enum import Enum, unique


@unique
class ModelType(Enum):
    """模型类别枚举"""
    PB = 'PB'
    ONNX = 'ONNX'
    TFLITE = 'TFLITE'


@unique
class DatasetType(Enum):
    """数据集类别枚举"""
    Directory = 'Directory'
    TFRecords = 'TFRecords'


@unique
class LabelFrom(Enum):
    """标签来源枚举"""
    XML = 'XML'
    LMDB = 'LMDB'
    FileName = 'FileName'
    TXT = 'TXT'


@unique
class LossFunction(Enum):
    """损失函数枚举"""
    CrossEntropy = 'CrossEntropy'
    CTC = 'CTC'


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
    ResNetTiny = 'ResNetTiny'
    ResNet50 = 'ResNet50'
    DenseNet = 'DenseNet'
    MobileNetV2 = 'MobileNetV2'


@unique
class RecurrentNetwork(Enum):
    """循环层枚举"""
    NoRecurrent = 'NoRecurrent'
    GRU = 'GRU'
    BiGRU = 'BiGRU'
    GRUcuDNN = 'GRUcuDNN'
    LSTM = 'LSTM'
    BiLSTM = 'BiLSTM'
    LSTMcuDNN = 'LSTMcuDNN'
    BiLSTMcuDNN = 'BiLSTMcuDNN'


@unique
class Optimizer(Enum):
    """优化器枚举"""
    RAdam = 'RAdam'
    Adam = 'Adam'
    Momentum = 'Momentum'
    AdaBound = 'AdaBound'
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
    FLOAT = 'FLOAT'
    CHS_3500 = 'CHS_3500'
    ALPHANUMERIC_CHS_3500_LOWER = 'ALPHANUMERIC_CHS_3500_LOWER'

