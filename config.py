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
CHAR_REPLACE = cf_model['Model'].get('CharReplace')
CHAR_REPLACE = CHAR_REPLACE if CHAR_REPLACE else {}
CHAR_SET_LEN = len(GEN_CHAR_SET)

"""MODEL"""
NEU_NAME = cf_system['System'].get('NeuralNet')
NEU_NAME = NEU_NAME if NEU_NAME else 'CNN+LSTM+CTC'
TARGET_MODEL = cf_model['Model'].get('ModelName')
IMAGE_HEIGHT = cf_model['Model'].get('ImageHeight')
IMAGE_WIDTH = cf_model['Model'].get('ImageWidth')

"""CNN"""
CNN_STRUCTURE = cf_model.get(NEU_NAME).get('CNN')
FILTERS = [1] + [i['Convolution'] for i in CNN_STRUCTURE]
CONV_KSIZE = [i['ConvCoreSize'] for i in CNN_STRUCTURE]
CONV_STRIDES = [i['ConvStrides'] for i in CNN_STRUCTURE]
POOL_STRIDES = [i['PoolStrides'] for i in CNN_STRUCTURE]
POOL_KSIZE = [i['PoolWindowSize'] for i in CNN_STRUCTURE]

"""LSTM"""
LSTM_STRUCTURE = cf_model.get(NEU_NAME).get('LSTM')
OUT_CHANNEL = CNN_STRUCTURE[-1].get('Convolution')
NUM_HIDDEN = LSTM_STRUCTURE.get('HiddenNum')
OUTPUT_KEEP_PROB = LSTM_STRUCTURE.get('KeepProb')

LEAKINESS = 0.01
NUM_CLASSES = CHAR_SET_LEN + 1

"""OPTIMIZER"""
# - The exponential decay rate for the 1st moment estimates.
BATE1 = 0.9
# - The exponential decay rate for the 2nd moment estimates.
BATE2 = 0.999

MODEL_TAG = '{}.model'.format(TARGET_MODEL)
CHECKPOINT_TAG = 'checkpoint'
SAVE_MODEL = os.path.join(MODEL_PATH, MODEL_TAG)
SAVE_CHECKPOINT = os.path.join(MODEL_PATH, CHECKPOINT_TAG)

"""SYSTEM"""
GPU_USAGE = cf_system['System'].get('DeviceUsage')

"""PATH & LABEL"""
TEST_PATH = cf_system['System'].get('TestPath')
TEST_REGEX = cf_system['System'].get('TestRegex')
TEST_REGEX = TEST_REGEX if TEST_REGEX else ".*?(?=_.*\.)"
TRAINS_PATH = cf_system['System'].get('TrainsPath')
TRAINS_REGEX = cf_system['System'].get('TrainRegex')
TRAINS_REGEX = TRAINS_REGEX if TRAINS_REGEX else ".*?(?=_.*\.)"
TEST_SET_NUM = cf_system['System'].get('TestSetNum')
TEST_SET_NUM = TEST_SET_NUM if TEST_SET_NUM else 1000
HAS_TEST_SET = TEST_PATH and os.path.exists(TEST_PATH)

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

"""PRETREATMENT"""
BINARYZATION = cf_model['Pretreatment'].get('Binaryzation')
SMOOTH = cf_model['Pretreatment'].get('Smoothing')
BLUR = cf_model['Pretreatment'].get('Blur')

"""COMPILE_MODEL"""
COMPILE_MODEL_PATH = os.path.join(MODEL_PATH, '{}.pb'.format(TARGET_MODEL))
QUANTIZED_MODEL_PATH = os.path.join(MODEL_PATH, 'quantized_{}.pb'.format(TARGET_MODEL))


def _checkpoint(_name, _path):
    file_list = os.listdir(_path)
    checkpoint = ['"{}"'.format(i.split(".meta")[0]) for i in file_list if i.startswith(_name) and i.endswith('.meta')]
    if not checkpoint:
        return None
    _checkpoint_step = [int(re.search('(?<=model-).*?(?=")', i).group()) for i in checkpoint]
    return checkpoint[_checkpoint_step.index(max(_checkpoint_step))]


def init():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

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
    print('NEURAL NETWORK: {}'.format(NEU_NAME))

    print('---------------------------------------------------------------------------------')
