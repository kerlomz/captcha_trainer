#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>

import os
import re
import sys
import yaml
import random
import platform
import PIL.Image as pilImage
from os.path import join
from character import *
from exception import *

PROJECT_PATH = os.path.dirname(os.path.realpath(sys.argv[0]))
# PROJECT_PARENT_PATH = os.path.abspath(os.path.join(PROJECT_PATH, os.path.pardir))

SYS_CONFIG_NAME = 'config.yaml'
MODEL_CONFIG_NAME = 'model.yaml'
PLATFORM = platform.system()

PATH_SPLIT = "\\" if PLATFORM == "Windows" else "/"
# For Service
SYS_CONFIG_PATH = os.path.join(PROJECT_PATH, SYS_CONFIG_NAME)
MODEL_PATH = os.path.join(PROJECT_PATH, 'model')

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

MODEL_CONFIG_PATH = os.path.join(PROJECT_PATH, MODEL_CONFIG_NAME)


if not os.path.exists(SYS_CONFIG_PATH):
    exception('Configuration File "{}" No Found.'.format(SYS_CONFIG_NAME))

if not os.path.exists(MODEL_CONFIG_PATH):
    exception('Configuration File "{}" No Found.'.format(MODEL_CONFIG_NAME))

with open(SYS_CONFIG_PATH, 'r', encoding="utf-8") as sys_fp:
    sys_stream = sys_fp.read()
    cf_system = yaml.load(sys_stream)

with open(MODEL_CONFIG_PATH, 'r', encoding="utf-8") as sys_fp:
    sys_stream = sys_fp.read()
    cf_model = yaml.load(sys_stream)

LANGUAGE = cf_system['System'].get('Language')
LANGUAGE = LANGUAGE if LANGUAGE else 'en-US'


def char_set(_name):
    if isinstance(_name, list):
        return _name
    if _name == 'NUMERIC':
        return NUMBER
    elif _name == 'ALPHANUMERIC':
        return NUMBER + ALPHA_LOWER + ALPHA_UPPER
    elif _name == 'ALPHANUMERIC_LOWER':
        return NUMBER + ALPHA_LOWER
    elif _name == 'ALPHANUMERIC_UPPER':
        return NUMBER + ALPHA_UPPER
    else:
        return NUMBER + ALPHA_LOWER + ALPHA_UPPER


def fetch_file_list(path):
    file_list = os.listdir(path)
    if len(file_list) < 50:
        exception("Insufficient Sample!")
    group = [os.path.join(path, image_file) for image_file in file_list]
    random.shuffle(group)
    return group


TARGET_MODEL = cf_model['Model'].get('ModelName')

CHAR_SET = cf_model['Model'].get('CharSet')
GEN_CHAR_SET = char_set(CHAR_SET)
CHAR_SET_LEN = len(GEN_CHAR_SET)

NEU_NAME = cf_system['System'].get('NeuralNet')

FILTERS = cf_model.get('DenseNet').get('Filters')

CONV_NEU_NUMS = cf_model.get('CNNNet').get('Convolution')
FULL_LAYER_FEATURE_NUM = cf_model['CNNNet'].get('FullConnect')
CONV_CORE_SIZE = cf_model.get('CNNNet').get('ConvCoreSize')

NEU_LAYER_NUM = len(CONV_NEU_NUMS)
MAX_POOL_NUM = NEU_LAYER_NUM

POOL_SIZE = [2, 2]
CONV_STRIDES = [1, 1, 1, 1]
POOL_STRIDES = [1, 2, 2, 1]
PADDING = 'SAME'


MODEL_TAG = '{}.model'.format(TARGET_MODEL)
CHECKPOINT_TAG = 'checkpoint'
SAVE_MODEL = join(MODEL_PATH, MODEL_TAG)
SAVE_CHECKPOINT = join(MODEL_PATH, CHECKPOINT_TAG)

DEVICE = cf_system['System'].get('Device')
DEVICE = DEVICE if DEVICE else "cpu:0"

TEST_PATH = cf_system['System'].get('TestPath')
TEST_REGEX = cf_system['System'].get('TestRegex')
TEST_REGEX = TEST_REGEX if TEST_REGEX else ".*?(?=_.*\.)"


TRAINS_PATH = cf_system['System'].get('TrainsPath')
TRAINS_REGEX = cf_system['System'].get('TrainRegex')
TRAINS_REGEX = TRAINS_REGEX if TRAINS_REGEX else ".*?(?=_.*\.)"

TRAINS_SAVE_STEP = cf_system['Trains'].get('SavedStep')

TRAINS_END_ACC = cf_system['Trains'].get('EndAcc')
TRAINS_END_STEP = cf_system['Trains'].get('EndStep')
TRAINS_LEARNING_RATE = cf_system['Trains'].get('LearningRate')
TRAINS_TEST_NUM = cf_system['Trains'].get('TestNum')


_TEST_GROUP = fetch_file_list(TEST_PATH)
_TRAIN_GROUP = fetch_file_list(TRAINS_PATH)

_IMAGE_PATH = _TEST_GROUP[random.randint(0, len(_TEST_GROUP) - 1)]
_TEST_IMAGE_SIZE = pilImage.open(_IMAGE_PATH).size
_TRAIN_IMAGE_SIZE = pilImage.open(_TRAIN_GROUP[0]).size
if _TEST_IMAGE_SIZE != _TRAIN_IMAGE_SIZE:
    exception("The image size of the test set must match the training set")

TEST_SAMPLE_LABEL = re.search(TEST_REGEX, _IMAGE_PATH.split(PATH_SPLIT)[-1]).group()

MAX_CAPTCHA_LEN = cf_model['Model'].get('CharLength')
MAX_CAPTCHA_LEN = MAX_CAPTCHA_LEN if MAX_CAPTCHA_LEN else len(TEST_SAMPLE_LABEL)

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
IMAGE_WIDTH = RESIZE[0] if RESIZE else _TEST_IMAGE_SIZE[0] * MAGNIFICATION
IMAGE_HEIGHT = RESIZE[1] if RESIZE else _TEST_IMAGE_SIZE[1] * MAGNIFICATION


def checkpoint(_name, _path):
    file_list = os.listdir(_path)
    _checkpoint = ['"{}"'.format(i.split(".meta")[0]) for i in file_list if i.startswith(_name) and i.endswith('.meta')]
    if not _checkpoint:
        return None
    _checkpoint_step = [int(re.search('(?<=model-).*?(?=")', i).group()) for i in _checkpoint]
    return _checkpoint[_checkpoint_step.index(max(_checkpoint_step))]


MODEL_FILE = checkpoint(TARGET_MODEL, MODEL_PATH)
# COMPILE_TRAINS_PATH = os.path.join(MODEL_PATH, '{}.tfrecords'.format(TARGET_MODEL))
COMPILE_MODEL_PATH = os.path.join(MODEL_PATH, '{}.pb'.format(TARGET_MODEL))
CHECKPOINT = 'model_checkpoint_path: {}\nall_model_checkpoint_paths: {}'.format(MODEL_FILE, MODEL_FILE)
with open(SAVE_CHECKPOINT, 'w') as f:
    f.write(CHECKPOINT)

print('Loading Configuration...')
print('---------------------------------------------------------------------------------')
print('CURRENT_CHECKPOINT:', checkpoint(TARGET_MODEL, MODEL_PATH))
# print("PROJECT_PARENT_PATH", PROJECT_PARENT_PATH)
print("PROJECT_PATH", PROJECT_PATH)
print('MODEL_PATH:', SAVE_MODEL)
print('COMPILE_MODEL_PATH:', COMPILE_MODEL_PATH)
print('CHAR_SET_LEN:', CHAR_SET_LEN)
print('IMAGE_WIDTH: {}, IMAGE_HEIGHT: {}{}'.format(IMAGE_WIDTH, IMAGE_HEIGHT, ", MAGNIFICATION: {}".format(MAGNIFICATION) if MAGNIFICATION and not RESIZE else ""))
print('IMAGE_ORIGINAL_COLOR: {}'.format(IMAGE_ORIGINAL_COLOR))
print("MAX_CAPTCHA_LEN", MAX_CAPTCHA_LEN)
print('NEURAL NETWORK: {}'.format(NEU_NAME))
if NEU_NAME == 'DenseNet':
    print('FILTERS: {}'.format(FILTERS))
else:
    print('{} LAYER CONV: {}, FULL_CONNECT: {}'.format(NEU_LAYER_NUM, CONV_NEU_NUMS, FULL_LAYER_FEATURE_NUM))
print('---------------------------------------------------------------------------------')
