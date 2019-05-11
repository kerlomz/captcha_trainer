#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
from enum import Enum, unique


@unique
class RunMode(Enum):
    Test = 'test'
    Trains = 'trains'
    Predict = 'predict'


@unique
class CNNNetwork(Enum):
    CNN5 = 'CNN5'
    ResNet = 'ResNet'
    DenseNet = 'DenseNet'


@unique
class RecurrentNetwork(Enum):
    LSTM = 'LSTM'
    BLSTM = 'BLSTM'
    SRU = 'SRU'
    BSRU = 'BSRU'
    GRU = 'GRU'


@unique
class Optimizer(Enum):
    AdaBound = 'AdaBound'
    Adam = 'Adam'
    Momentum = 'Momentum'
    SGD = 'SGD'
    AdaGrad = 'AdaGrad'
    RMSProp = 'RMSProp'


@unique
class SimpleCharset(Enum):
    NUMERIC = 'NUMBER'
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

