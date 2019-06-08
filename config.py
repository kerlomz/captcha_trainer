#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>

import os
import platform
import re
import yaml

from character import *
from constants import *
from exception import exception, ConfigException

# Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
# If you have a GPU, you shouldn't care about AVX support.
# Just disables the warning, doesn't enable AVX/FMA
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
PROJECT_PATH = "."
IGNORE_FILES = ['.DS_Store']


NETWORK_MAP = {
    'CNN5': CNNNetwork.CNN5,
    'ResNet': CNNNetwork.ResNet,
    'DenseNet': CNNNetwork.DenseNet,
    'LSTM': RecurrentNetwork.LSTM,
    'BLSTM': RecurrentNetwork.BLSTM,
    'SRU': RecurrentNetwork.SRU,
    'BSRU': RecurrentNetwork.BSRU,
    'GRU': RecurrentNetwork.GRU,
}


OPTIMIZER_MAP = {
    'AdaBound': Optimizer.AdaBound,
    'Adam': Optimizer.Adam,
    'Momentum': Optimizer.Momentum,
    'SGD': Optimizer.SGD,
    'AdaGrad': Optimizer.AdaGrad,
    'RMSProp': Optimizer.RMSProp
}

PLATFORM = platform.system()

# SYS_CONFIG_DEMO_NAME = 'config_demo.yaml'
MODEL_CONFIG_DEMO_NAME = 'model_demo.yaml'
# SYS_CONFIG_NAME = 'config.yaml'
MODEL_CONFIG_NAME = 'model.yaml'
MODEL_PATH = os.path.join(PROJECT_PATH, 'model')
OUTPUT_PATH = os.path.join(PROJECT_PATH, 'out')
TFRECORDS_DIR = os.path.join(PROJECT_PATH, 'dataset')

PATH_SPLIT = "\\" if PLATFORM == "Windows" else "/"

# SYS_CONFIG_PATH = os.path.join(PROJECT_PATH, SYS_CONFIG_NAME)
# SYS_CONFIG_PATH = SYS_CONFIG_PATH if os.path.exists(SYS_CONFIG_PATH) else os.path.join("../", SYS_CONFIG_NAME)

MODEL_CONFIG_PATH = os.path.join(PROJECT_PATH, MODEL_CONFIG_NAME)
MODEL_CONFIG_PATH = MODEL_CONFIG_PATH if os.path.exists(MODEL_CONFIG_PATH) else os.path.join("../", MODEL_CONFIG_NAME)

# with open(SYS_CONFIG_PATH, 'r', encoding="utf-8") as sys_fp:
#     sys_stream = sys_fp.read()
#     cf_system = yaml.load(sys_stream, Loader=yaml.SafeLoader)

with open(MODEL_CONFIG_PATH, 'r', encoding="utf-8") as sys_fp:
    sys_stream = sys_fp.read()
    cf_model = yaml.load(sys_stream, Loader=yaml.SafeLoader)


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
CASE_SENSITIVE = cf_model['Model'].get('CaseSensitive')
CASE_SENSITIVE = CASE_SENSITIVE if CASE_SENSITIVE is not None else True

"""MODEL"""
# NEU_NETWORK = cf_system['NeuralNet']
TARGET_MODEL = cf_model['Model'].get('ModelName')
IMAGE_HEIGHT = cf_model['Model'].get('ImageHeight')
IMAGE_WIDTH = cf_model['Model'].get('ImageWidth')
IMAGE_CHANNEL = cf_model['Model'].get('ImageChannel')
IMAGE_CHANNEL = IMAGE_CHANNEL if IMAGE_CHANNEL else 1
MULTI_SHAPE = False


"""NEURAL NETWORK"""
NEU_CNN = cf_model['NeuralNet'].get('CNNNetwork')
NEU_CNN = NEU_CNN if NEU_CNN else 'CNN5'
NEU_RECURRENT = cf_model['NeuralNet'].get('RecurrentNetwork')
NEU_RECURRENT = NEU_RECURRENT if NEU_RECURRENT else 'BLSTM'
NUM_HIDDEN = cf_model['NeuralNet'].get('HiddenNum')
OUTPUT_KEEP_PROB = cf_model['NeuralNet'].get('KeepProb')
LSTM_LAYER_NUM = 2
NEU_OPTIMIZER = cf_model['NeuralNet'].get('Optimizer')
NEU_OPTIMIZER = NEU_OPTIMIZER if NEU_OPTIMIZER else 'AdaBound'
PREPROCESS_COLLAPSE_REPEATED = cf_model['NeuralNet'].get('PreprocessCollapseRepeated')
PREPROCESS_COLLAPSE_REPEATED = PREPROCESS_COLLAPSE_REPEATED if PREPROCESS_COLLAPSE_REPEATED is not None else False
CTC_MERGE_REPEATED = cf_model['NeuralNet'].get('CTCMergeRepeated')
CTC_MERGE_REPEATED = CTC_MERGE_REPEATED if CTC_MERGE_REPEATED is not None else True
CTC_BEAM_WIDTH = cf_model['NeuralNet'].get('CTCBeamWidth')
CTC_BEAM_WIDTH = CTC_BEAM_WIDTH if CTC_BEAM_WIDTH is not None else 1
CTC_TOP_PATHS = cf_model['NeuralNet'].get('CTCTopPaths')
CTC_TOP_PATHS = CTC_TOP_PATHS if CTC_TOP_PATHS is not None else 1
CTC_LOSS_TIME_MAJOR = True
WARP_CTC = cf_model['NeuralNet'].get('WarpCTC')
WARP_CTC = WARP_CTC if WARP_CTC is not None else False


LEAKINESS = 0.01
NUM_CLASSES = CHAR_SET_LEN + 2

MODEL_TAG = '{}.model'.format(TARGET_MODEL)
CHECKPOINT_TAG = 'checkpoint'
SAVE_MODEL = os.path.join(MODEL_PATH, MODEL_TAG)
SAVE_CHECKPOINT = os.path.join(MODEL_PATH, CHECKPOINT_TAG)

"""SYSTEM"""
GPU_USAGE = cf_model['System'].get('DeviceUsage')

"""PATH & LABEL"""
TRAIN_PATH_IN_MODEL = cf_model.get('Trains')


TRAINS_PATH = cf_model['Trains'].get('TrainsPath')
TEST_PATH = cf_model['Trains'].get('TestPath')
DATASET_PATH = cf_model['Trains'].get('DatasetPath')

TRAINS_REGEX = cf_model['Trains'].get('TrainRegex')
TRAINS_REGEX = TRAINS_REGEX if TRAINS_REGEX else ".*?(?=_)"

TEST_REGEX = cf_model['Trains'].get('TestRegex')
TEST_REGEX = TEST_REGEX if TEST_REGEX else (TRAINS_REGEX if TRAINS_REGEX else ".*?(?=_)")

TEST_SET_NUM = cf_model['Trains'].get('TestSetNum')
TEST_SET_NUM = TEST_SET_NUM if TEST_SET_NUM else 1000
HAS_TEST_SET = TEST_PATH and (os.path.exists(TEST_PATH) if isinstance(TEST_PATH, str) else True)

SPLIT_DATASET = not TEST_PATH
TEST_USE_TFRECORDS = isinstance(TEST_PATH, str) and TEST_PATH.endswith("tfrecords")
TRAINS_USE_TFRECORDS = isinstance(TRAINS_PATH, str) and TRAINS_PATH.endswith("tfrecords")

"""TRAINS"""
TRAINS_SAVE_STEPS = cf_model['Trains'].get('SavedSteps')
TRAINS_VALIDATION_STEPS = cf_model['Trains'].get('ValidationSteps')
TRAINS_END_ACC = cf_model['Trains'].get('EndAcc')
TRAINS_END_COST = cf_model['Trains'].get('EndCost')
TRAINS_END_COST = TRAINS_END_COST if TRAINS_END_COST else 1
TRAINS_END_EPOCHS = cf_model['Trains'].get('EndEpochs')
TRAINS_LEARNING_RATE = cf_model['Trains'].get('LearningRate')
DECAY_RATE = cf_model['Trains'].get('DecayRate')
DECAY_RATE = DECAY_RATE if DECAY_RATE else 0.98
DECAY_STEPS = cf_model['Trains'].get('DecaySteps')
DECAY_STEPS = DECAY_STEPS if DECAY_STEPS else 10000
BATCH_SIZE = cf_model['Trains'].get('BatchSize')
BATCH_SIZE = BATCH_SIZE if BATCH_SIZE else 64
TEST_BATCH_SIZE = cf_model['Trains'].get('TestBatchSize')
TEST_BATCH_SIZE = TEST_BATCH_SIZE if TEST_BATCH_SIZE else 300
MOMENTUM = 0.9

"""PRETREATMENT"""
BINARYZATION = cf_model['Pretreatment'].get('Binaryzation')
SMOOTH = cf_model['Pretreatment'].get('Smoothing')
BLUR = cf_model['Pretreatment'].get('Blur')
REPLACE_TRANSPARENT = cf_model['Pretreatment'].get('ReplaceTransparent')
RESIZE = cf_model['Pretreatment'].get('Resize')
RESIZE = RESIZE if RESIZE else [IMAGE_WIDTH, IMAGE_HEIGHT]

"""COMPILE_MODEL"""
COMPILE_MODEL_PATH = os.path.join(OUTPUT_PATH, '{}.pb'.format(TARGET_MODEL))
QUANTIZED_MODEL_PATH = os.path.join(MODEL_PATH, 'quantized_{}.pb'.format(TARGET_MODEL))


def _checkpoint(_name, _path):
    file_list = os.listdir(_path)
    checkpoint = ['"{}"'.format(i.split(".meta")[0]) for i in file_list if
                  _name + ".model" in i and i.endswith('.meta')]
    if not checkpoint:
        return None
    _checkpoint_step = [int(re.search('(?<=model-).*?(?=")', i).group()) for i in checkpoint]
    return checkpoint[_checkpoint_step.index(max(_checkpoint_step))]


def init():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    # if not os.path.exists(SYS_CONFIG_PATH):
    #     exception(
    #         'Configuration File "{}" No Found. '
    #         'If it is used for the first time, please copy one from {} as {}'.format(
    #             SYS_CONFIG_NAME,
    #             SYS_CONFIG_DEMO_NAME,
    #             SYS_CONFIG_NAME
    #         ), ConfigException.SYS_CONFIG_PATH_NOT_EXIST
    #     )

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


if '../' not in MODEL_CONFIG_PATH:
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
    print('NEURAL NETWORK: {}'.format(cf_model['NeuralNet']))

    print('---------------------------------------------------------------------------------')