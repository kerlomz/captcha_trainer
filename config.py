#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>

import os
import re
import yaml
import platform
from character import *
from exception import exception, ConfigException

# Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
# If you have a GPU, you shouldn't care about AVX support.
# Just disables the warning, doesn't enable AVX/FMA
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
PROJECT_PATH = "."


class RunMode(object):
    Test = 'test'
    Trains = 'trains'
    Predict = 'predict'


class CNNNetwork(object):
    CNN5 = 'CNN5'
    DenseNet = 'DenseNet'


class RecurrentNetwork:
    LSTM = 'LSTM'
    BLSTM = 'BLSTM'


NETWORK_MAP = {
    'CNN5': CNNNetwork.CNN5,
    'DenseNet': CNNNetwork.DenseNet,
    'LSTM': RecurrentNetwork.LSTM,
    'BLSTM': RecurrentNetwork.BLSTM,
}


TFRECORDS_NAME_MAP = {
    RunMode.Trains: 'trains',
    RunMode.Test: 'test'
}

PLATFORM = platform.system()

SYS_CONFIG_DEMO_NAME = 'config_demo.yaml'
MODEL_CONFIG_DEMO_NAME = 'model_demo.yaml'
SYS_CONFIG_NAME = 'config.yaml'
MODEL_CONFIG_NAME = 'model.yaml'
MODEL_PATH = os.path.join(PROJECT_PATH, 'model')
OUTPUT_PATH = os.path.join(PROJECT_PATH, 'out')
TFRECORDS_DIR = os.path.join(PROJECT_PATH, 'dataset')

PATH_SPLIT = "\\" if PLATFORM == "Windows" else "/"

SYS_CONFIG_PATH = os.path.join(PROJECT_PATH, SYS_CONFIG_NAME)
SYS_CONFIG_PATH = SYS_CONFIG_PATH if os.path.exists(SYS_CONFIG_PATH) else os.path.join("../", SYS_CONFIG_NAME)

MODEL_CONFIG_PATH = os.path.join(PROJECT_PATH, MODEL_CONFIG_NAME)
MODEL_CONFIG_PATH = MODEL_CONFIG_PATH if os.path.exists(MODEL_CONFIG_PATH) else os.path.join("../", MODEL_CONFIG_NAME)

with open(SYS_CONFIG_PATH, 'r', encoding="utf-8") as sys_fp:
    sys_stream = sys_fp.read()
    cf_system = yaml.load(sys_stream)

with open(MODEL_CONFIG_PATH, 'r', encoding="utf-8") as sys_fp:
    sys_stream = sys_fp.read()
    cf_model = yaml.load(sys_stream)


def char_set(_type):
    if isinstance(_type, list):
        return _type
    if isinstance(_type, str):
        return SIMPLE_CHAR_SET.get(_type) if _type in SIMPLE_CHAR_SET.keys() else ConfigException.CHAR_SET_NOT_EXIST
    exception(
        "Character set configuration error, customized character set should be list type",
        ConfigException.CHAR_SET_INCORRECT
    )


"""CHARSET"""
CHAR_SET = cf_model['Model'].get('CharSet')
CHAR_EXCLUDE = cf_model['Model'].get('CharExclude')
GEN_CHAR_SET = [i for i in char_set(CHAR_SET) if i not in CHAR_EXCLUDE]
GEN_CHAR_SET = [''] + GEN_CHAR_SET

# fixed Not enough time for target transition sequence
# GEN_CHAR_SET = SPACE_TOKEN + GEN_CHAR_SET
CHAR_REPLACE = cf_model['Model'].get('CharReplace')
CHAR_REPLACE = CHAR_REPLACE if CHAR_REPLACE else {}
CHAR_SET_LEN = len(GEN_CHAR_SET)

"""MODEL"""
# NEU_NETWORK = cf_system['NeuralNet']
TARGET_MODEL = cf_model['Model'].get('ModelName')
IMAGE_HEIGHT = cf_model['Model'].get('ImageHeight')
IMAGE_WIDTH = cf_model['Model'].get('ImageWidth')

"""NEURAL NETWORK"""
NEU_CNN = cf_system['NeuralNet'].get('CNNNetwork')
NEU_CNN = NEU_CNN if NEU_CNN else 'CNN5'
NEU_RECURRENT = cf_system['NeuralNet'].get('RecurrentNetwork')
NEU_RECURRENT = NEU_RECURRENT if NEU_RECURRENT else 'BLSTM'
NUM_HIDDEN = cf_system['NeuralNet'].get('HiddenNum')
OUTPUT_KEEP_PROB = cf_system['NeuralNet'].get('KeepProb')
LSTM_LAYER_NUM = 2

LEAKINESS = 0.01
NUM_CLASSES = CHAR_SET_LEN + 2

MODEL_TAG = '{}.model'.format(TARGET_MODEL)
CHECKPOINT_TAG = 'checkpoint'
SAVE_MODEL = os.path.join(MODEL_PATH, MODEL_TAG)
SAVE_CHECKPOINT = os.path.join(MODEL_PATH, CHECKPOINT_TAG)

"""SYSTEM"""
GPU_USAGE = cf_system['System'].get('DeviceUsage')

"""PATH & LABEL"""
TRAIN_PATH_IN_MODEL = cf_model.get('Trains')

if TRAIN_PATH_IN_MODEL:
    TRAINS_PATH = cf_model['Trains'].get('TrainsPath')
    TEST_PATH = cf_model['Trains'].get('TestPath')
else:
    TRAINS_PATH = cf_system['System'].get('TrainsPath')
    TEST_PATH = cf_system['System'].get('TestPath')

TEST_REGEX = cf_system['System'].get('TestRegex')
TEST_REGEX = TEST_REGEX if TEST_REGEX else ".*?(?=_.*\.)"

TRAINS_REGEX = cf_system['System'].get('TrainRegex')
TRAINS_REGEX = TRAINS_REGEX if TRAINS_REGEX else ".*?(?=_.*\.)"
TEST_SET_NUM = cf_system['System'].get('TestSetNum')
TEST_SET_NUM = TEST_SET_NUM if TEST_SET_NUM else 1000
HAS_TEST_SET = TEST_PATH and (os.path.exists(TEST_PATH) if isinstance(TEST_PATH, str) else True)

SPLIT_DATASET = not TEST_PATH
TEST_USE_TFRECORDS = isinstance(TEST_PATH, str) and TEST_PATH.endswith("tfrecords")
TRAINS_USE_TFRECORDS = isinstance(TRAINS_PATH, str) and TRAINS_PATH.endswith("tfrecords")

"""TRAINS"""
TRAINS_SAVE_STEPS = cf_system['Trains'].get('SavedSteps')
TRAINS_VALIDATION_STEPS = cf_system['Trains'].get('ValidationSteps')
TRAINS_END_ACC = cf_system['Trains'].get('EndAcc')
TRAINS_END_EPOCHS = cf_system['Trains'].get('EndEpochs')
TRAINS_LEARNING_RATE = cf_system['Trains'].get('LearningRate')
DECAY_RATE = cf_system['Trains'].get('DecayRate')
DECAY_STEPS = cf_system['Trains'].get('DecaySteps')
BATCH_SIZE = cf_system['Trains'].get('BatchSize')
TEST_BATCH_SIZE = cf_system['Trains'].get('TestBatchSize')
TEST_BATCH_SIZE = TEST_BATCH_SIZE if TEST_BATCH_SIZE else 200
MOMENTUM = 0.9

"""PRETREATMENT"""
BINARYZATION = cf_model['Pretreatment'].get('Binaryzation')
SMOOTH = cf_model['Pretreatment'].get('Smoothing')
BLUR = cf_model['Pretreatment'].get('Blur')
RESIZE = cf_model['Pretreatment'].get('Resize')
RESIZE = RESIZE if RESIZE else [IMAGE_WIDTH, IMAGE_HEIGHT]

"""COMPILE_MODEL"""
COMPILE_MODEL_PATH = os.path.join(OUTPUT_PATH, '{}.pb'.format(TARGET_MODEL))
QUANTIZED_MODEL_PATH = os.path.join(MODEL_PATH, 'quantized_{}.pb'.format(TARGET_MODEL))


def _checkpoint(_name, _path):
    file_list = os.listdir(_path)
    checkpoint = ['"{}"'.format(i.split(".meta")[0]) for i in file_list if _name + ".model" in i and i.endswith('.meta')]
    if not checkpoint:
        return None
    _checkpoint_step = [int(re.search('(?<=model-).*?(?=")', i).group()) for i in checkpoint]
    return checkpoint[_checkpoint_step.index(max(_checkpoint_step))]


def init():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    if not os.path.exists(SYS_CONFIG_PATH):
        exception(
            'Configuration File "{}" No Found. '
            'If it is used for the first time, please copy one from {} as {}'.format(
                SYS_CONFIG_NAME,
                SYS_CONFIG_DEMO_NAME,
                SYS_CONFIG_NAME
            ), ConfigException.SYS_CONFIG_PATH_NOT_EXIST
        )

    if not os.path.exists(MODEL_CONFIG_PATH):
        exception(
            'Configuration File "{}" No Found. '
            'If it is used for the first time, please copy one from {} as {}'.format(
                MODEL_CONFIG_NAME,
                MODEL_CONFIG_DEMO_NAME,
                MODEL_CONFIG_NAME
            ), ConfigException.MODEL_CONFIG_PATH_NOT_EXIST
        )

    if not isinstance(CHAR_EXCLUDE, list):
        exception("\"CharExclude\" should be a list")

    if GEN_CHAR_SET == ConfigException.CHAR_SET_NOT_EXIST:
        exception(
            "The character set type does not exist, there is no character set named {}".format(CHAR_SET),
            ConfigException.CHAR_SET_NOT_EXIST
        )

    model_file = _checkpoint(TARGET_MODEL, MODEL_PATH)
    checkpoint = 'model_checkpoint_path: {}\nall_model_checkpoint_paths: {}'.format(model_file, model_file)
    with open(SAVE_CHECKPOINT, 'w') as f:
        f.write(checkpoint)


if '../' not in SYS_CONFIG_PATH:
    print('Loading Configuration...')
    print('---------------------------------------------------------------------------------')
    print("PROJECT_PATH", PROJECT_PATH)
    print('MODEL_PATH:', SAVE_MODEL)
    print('COMPILE_MODEL_PATH:', COMPILE_MODEL_PATH)
    print('CHAR_SET_LEN:', CHAR_SET_LEN)
    print('CHAR_REPLACE: {}'.format(CHAR_REPLACE))
    print('IMAGE_WIDTH: {}, IMAGE_HEIGHT: {}'.format(
        IMAGE_WIDTH, IMAGE_HEIGHT)
    )
    print('NEURAL NETWORK: {}'.format(cf_system['NeuralNet']))

    print('---------------------------------------------------------------------------------')
