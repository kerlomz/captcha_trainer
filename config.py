#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>

import os
import re
import yaml
import random
import platform
import PIL.Image as pilImage
from character import *
from exception import exception, ConfigException

# Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
# If you have a GPU, you shouldn't care about AVX support.
# Just disables the warning, doesn't enable AVX/FMA
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
PROJECT_PATH = "."

PLATFORM = platform.system()

SYS_CONFIG_DEMO_NAME = 'config_demo.yaml'
MODEL_CONFIG_DEMO_NAME = 'model_demo.yaml'
SYS_CONFIG_NAME = 'config.yaml'
MODEL_CONFIG_NAME = 'model.yaml'
MODEL_PATH = os.path.join(PROJECT_PATH, 'model')

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


def parse_neural_structure(_net):
    layer = ""
    layer_structure = []
    layer_num = 1
    pre_input = 1
    for i in _net:
        key = list(i.keys())[0]
        val = list(i.values())[0]
        conv = {"index": layer_num, "input": pre_input, "output": val, "extra": []}
        if key == 'Convolution':
            layer += "\n - {} Layer: {} Layer-[{} * {}]".format(layer_num, key, val, val)
            layer_structure.append(conv)
            pre_input = val
            layer_num += 1
        if key == 'Pool':
            layer += ", {} Layer-{}".format(key, val)
            layer_structure[layer_num - 2]['extra'].append({"name": "pool", "window": val})
        if key == 'Optimization':
            layer += ", {} Layer".format(val)
            layer_structure[layer_num - 2]['extra'].append({"name": "dropout"})
    return layer[1:], layer_structure


def fetch_file_list(path):
    file_list = os.listdir(path)
    if len(file_list) < 200:
        exception("Insufficient Sample!", ConfigException.INSUFFICIENT_SAMPLE)
    group = [os.path.join(path, image_file) for image_file in file_list]
    random.shuffle(group)
    return group


TARGET_MODEL = cf_model['Model'].get('ModelName')

CHAR_SET = cf_model['Model'].get('CharSet')
CHAR_EXCLUDE = cf_model['Model'].get('CharExclude')

GEN_CHAR_SET = [i for i in char_set(CHAR_SET) if i not in CHAR_EXCLUDE]

CHAR_SET_LEN = len(GEN_CHAR_SET)

NEU_NAME = cf_system['System'].get('NeuralNet')

CONV_NEU_LAYER = cf_model.get('CNNNet').get('Layer')
CONV_NEU_LAYER_DESC, CONV_NEU_STRUCTURE = parse_neural_structure(CONV_NEU_LAYER)

FULL_LAYER_FEATURE_NUM = cf_model['CNNNet'].get('FullConnect')
CONV_CORE_SIZE = cf_model.get('CNNNet').get('ConvCoreSize')

NEU_LAYER_NUM = len(CONV_NEU_STRUCTURE)
MAX_POOL_NUM = len([i for i in CONV_NEU_LAYER if list(i.keys())[0] == 'Pool'])

CONV_STRIDES = [1, 1, 1, 1]
POOL_STRIDES = [1, 2, 2, 1]
PADDING = 'SAME'

MODEL_TAG = '{}.model'.format(TARGET_MODEL)
CHECKPOINT_TAG = 'checkpoint'
SAVE_MODEL = os.path.join(MODEL_PATH, MODEL_TAG)
SAVE_CHECKPOINT = os.path.join(MODEL_PATH, CHECKPOINT_TAG)

DEVICE = cf_system['System'].get('Device')
GPU_USAGE = cf_system['System'].get('DeviceUsage')

TEST_PATH = cf_system['System'].get('TestPath')
TEST_REGEX = cf_system['System'].get('TestRegex')
TEST_REGEX = TEST_REGEX if TEST_REGEX else ".*?(?=_.*\.)"

TRAINS_PATH = cf_system['System'].get('TrainsPath')
TRAINS_REGEX = cf_system['System'].get('TrainRegex')
TRAINS_REGEX = TRAINS_REGEX if TRAINS_REGEX else ".*?(?=_.*\.)"

TRAINS_SAVE_STEP = cf_system['Trains'].get('SavedStep')
COMPILE_ACC = cf_system['Trains'].get('CompileAcc')
TRAINS_END_ACC = cf_system['Trains'].get('EndAcc')
TRAINS_END_STEP = cf_system['Trains'].get('EndStep')
TRAINS_LEARNING_RATE = cf_system['Trains'].get('LearningRate')
TRAINS_TEST_NUM = cf_system['Trains'].get('TestNum')

_TEST_GROUP = fetch_file_list(TEST_PATH)
_TRAIN_GROUP = fetch_file_list(TRAINS_PATH)

IMAGE_CHANNEL = cf_model['Model'].get('ImageChannel')

MAGNIFICATION = cf_model['Pretreatment'].get('Magnification')
MAGNIFICATION = MAGNIFICATION if MAGNIFICATION and MAGNIFICATION > 0 and isinstance(MAGNIFICATION, int) else 1
IMAGE_ORIGINAL_COLOR = cf_model['Pretreatment'].get('OriginalColor')
BINARYZATION = cf_model['Pretreatment'].get('Binaryzation')
INVERT = cf_model['Pretreatment'].get('Invert')
SMOOTH = cf_model['Pretreatment'].get('Smoothing')
BLUR = cf_model['Pretreatment'].get('Blur')
RESIZE = cf_model['Pretreatment'].get('Resize')
RESIZE = tuple(RESIZE) if RESIZE else None

_IMAGE_PATH = _TEST_GROUP[random.randint(0, len(_TEST_GROUP) - 1)]
_TEST_IMAGE_SIZE = pilImage.open(_IMAGE_PATH).size
_TRAIN_IMAGE_SIZE = pilImage.open(_TRAIN_GROUP[0]).size

TEST_SAMPLE_LABEL = re.search(TEST_REGEX, _IMAGE_PATH.split(PATH_SPLIT)[-1]).group()

MAX_CAPTCHA_LEN = cf_model['Model'].get('CharLength')
MAX_CAPTCHA_LEN = MAX_CAPTCHA_LEN if MAX_CAPTCHA_LEN else len(TEST_SAMPLE_LABEL)
IMAGE_WIDTH = RESIZE[0] if RESIZE else _TEST_IMAGE_SIZE[0] * MAGNIFICATION
IMAGE_HEIGHT = RESIZE[1] if RESIZE else _TEST_IMAGE_SIZE[1] * MAGNIFICATION


def checkpoint(_name, _path):
    file_list = os.listdir(_path)
    _checkpoint = ['"{}"'.format(i.split(".meta")[0]) for i in file_list if i.startswith(_name) and i.endswith('.meta')]
    if not _checkpoint:
        return None
    _checkpoint_step = [int(re.search('(?<=model-).*?(?=")', i).group()) for i in _checkpoint]
    return _checkpoint[_checkpoint_step.index(max(_checkpoint_step))]


# COMPILE_TRAINS_PATH = os.path.join(MODEL_PATH, '{}.tfrecords'.format(TARGET_MODEL))
COMPILE_MODEL_PATH = os.path.join(MODEL_PATH, '{}.pb'.format(TARGET_MODEL))
TF_LITE_MODEL_PATH = os.path.join(MODEL_PATH, "{}.tflite".format(TARGET_MODEL))
QUANTIZED_MODEL_PATH = os.path.join(MODEL_PATH, 'quantized_{}.pb'.format(TARGET_MODEL))


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

    if _TEST_IMAGE_SIZE != _TRAIN_IMAGE_SIZE and not RESIZE:
        exception("The image size of the test set must match the training set")

    MODEL_FILE = checkpoint(TARGET_MODEL, MODEL_PATH)
    CHECKPOINT = 'model_checkpoint_path: {}\nall_model_checkpoint_paths: {}'.format(MODEL_FILE, MODEL_FILE)
    with open(SAVE_CHECKPOINT, 'w') as f:
        f.write(CHECKPOINT)


if '../' not in SYS_CONFIG_PATH:
    print('Loading Configuration...')
    print('---------------------------------------------------------------------------------')

    # print("PROJECT_PARENT_PATH", PROJECT_PARENT_PATH)
    print("PROJECT_PATH", PROJECT_PATH)
    print('MODEL_PATH:', SAVE_MODEL)
    print('COMPILE_MODEL_PATH:', COMPILE_MODEL_PATH)
    print('CHAR_SET_LEN:', CHAR_SET_LEN)
    print('IMAGE_WIDTH: {}, IMAGE_HEIGHT: {}{}'.format(
        IMAGE_WIDTH, IMAGE_HEIGHT, ", MAGNIFICATION: {}".format(
            MAGNIFICATION) if MAGNIFICATION and not RESIZE else "")
    )
    print('IMAGE_ORIGINAL_COLOR: {}'.format(IMAGE_ORIGINAL_COLOR))
    print("MAX_CAPTCHA_LEN", MAX_CAPTCHA_LEN)
    print('NEURAL NETWORK: {}'.format(NEU_NAME))
    print('{} LAYER CONV: \n{}\n - Full Connect Layer: {}'.format(
        NEU_LAYER_NUM, CONV_NEU_LAYER_DESC, FULL_LAYER_FEATURE_NUM
    ))

    print('---------------------------------------------------------------------------------')
