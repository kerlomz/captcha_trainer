#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import os
import json
import PIL.Image as pilImage
from constants import *

# - [ALPHANUMERIC, ALPHANUMERIC_LOWER, ALPHANUMERIC_UPPER,
# -- NUMERIC, ALPHABET_LOWER, ALPHABET_UPPER, ALPHABET, ALPHANUMERIC_LOWER_MIX_CHINESE_3500]
charset = SimpleCharset.ALPHANUMERIC_LOWER

cnn_network = CNNNetwork.CNN5
recurrent_network = RecurrentNetwork.BLSTM
optimizer = Optimizer.AdaBound

trains_path = [
    r"D:\TrainSet\***",
]

test_num = 500
hidden_num = 64
beam_width = 1
learning_rate = None

name_prefix = None
name_suffix = None
name_prefix = name_prefix if name_prefix else "tutorial"
name_suffix = '-' + str(name_suffix) if name_suffix else ''

model = """
# - requirement.txt  -  GPU: tensorflow-gpu, CPU: tensorflow
# - If you use the GPU version, you need to install some additional applications.
System:
  DeviceUsage: 0.7
  
# ModelName: Corresponding to the model file in the model directory,
# - such as YourModelName.pb, fill in YourModelName here.
# CharSet: Provides a default optional built-in solution:
# - [ALPHANUMERIC, ALPHANUMERIC_LOWER, ALPHANUMERIC_UPPER,
# -- NUMERIC, ALPHABET_LOWER, ALPHABET_UPPER, ALPHABET, ALPHANUMERIC_LOWER_MIX_CHINESE_3500]
# - Or you can use your own customized character set like: ['a', '1', '2'].
# CharMaxLength: Maximum length of characters， used for label padding.
# CharExclude: CharExclude should be a list, like: ['a', '1', '2']
# - which is convenient for users to freely combine character sets.
# - If you don't want to manually define the character set manually,
# - you can choose a built-in character set
# - and set the characters to be excluded by CharExclude parameter.
Model:
  Sites: [
  ]
  ModelName: @model_name
  ModelType: @size_str
  CharSet: @charset
  CharExclude: []
  CharReplace: {}
  ImageWidth: @width
  ImageHeight: @height

# Binaryzation: [-1: Off, >0 and < 255: On].
# Smoothing: [-1: Off, >0: On].
# Blur: [-1: Off, >0: On].
# Resize: [WIDTH, HEIGHT]
# - If the image size is too small, the training effect will be poor and you need to zoom in.
# ReplaceTransparent: [True, False]
# - True: Convert transparent images in RGBA format to opaque RGB format,
# - False: Keep the original image
Pretreatment:
  Binaryzation: -1
  Smoothing: -1
  Blur: -1
  Resize: @resize
  ReplaceTransparent: True

# CNNNetwork: [CNN5, ResNet, DenseNet]
# RecurrentNetwork: [BLSTM, LSTM, SRU, BSRU, GRU]
# - The recommended configuration is CNN5+BLSTM / ResNet+BLSTM
# HiddenNum: [64, 128, 256]
# - This parameter indicates the number of nodes used to remember and store past states.
# Optimizer: Loss function algorithm for calculating gradient.
# - [AdaBound, Adam, Momentum, SGD, AdaGrad, RMSProp]
NeuralNet:
  CNNNetwork: @cnn_network
  RecurrentNetwork: @recurrent_network
  HiddenNum: @hidden_num
  KeepProb: 0.98
  Optimizer: @optimizer
  PreprocessCollapseRepeated: False
  CTCMergeRepeated: True
  CTCBeamWidth: @beam_width
  CTCTopPaths: 1
  WarpCTC: False
  
# TrainsPath and TestPath: The local absolute path of your training and testing set.
# DatasetPath: Package a sample of the TFRecords format from this path.
# TrainRegex and TestRegex: Default matching apple_20181010121212.jpg file.
# - The Default is .*?(?=_.*\.)
# TestSetNum: This is an optional parameter that is used when you want to extract some of the test set
# - from the training set when you are not preparing the test set separately.
# SavedSteps: A Session.run() execution is called a Step,
# - Used to save training progress, Default value is 100.
# ValidationSteps: Used to calculate accuracy, Default value is 500.
# TestSetNum: The number of test sets, if an automatic allocation strategy is used (TestPath not set).
# EndAcc: Finish the training when the accuracy reaches [EndAcc*100]% and other conditions.
# EndCost: Finish the training when the cost reaches EndCost and other conditions.
# EndEpochs: Finish the training when the epoch is greater than the defined epoch and other conditions.
# BatchSize: Number of samples selected for one training step.
# TestBatchSize: Number of samples selected for one validation step.
# LearningRate: Recommended value[0.01: MomentumOptimizer/AdamOptimizer, 0.001: AdaBoundOptimizer]
Trains:
  TrainsPath: './dataset/@model_name_trains.tfrecords'
  TestPath: './dataset/@model_name_test.tfrecords'
  DatasetPath: @trains_path
  TrainRegex: '.*?(?=_)'
  TestSetNum: @test_num
  SavedSteps: 100
  ValidationSteps: 500
  EndAcc: 0.95
  EndCost: 0.1
  EndEpochs: 2
  BatchSize: 128
  TestBatchSize: 300
  LearningRate: @learning_rate
  DecayRate: 0.98
  DecaySteps: 10000
"""

trains_path = [i.replace("\\", "/") for i in trains_path]
file_name = os.listdir(trains_path[0])[0]
size = pilImage.open(os.path.join(trains_path[0], file_name)).size

width = size[0]
height = size[1]

size_str = "{}x{}".format(width, height)
if width > 160 or width < 120:
    r_height = int(height * 150 / width)
else:
    r_height = height
resize = "[{}, {}]".format(width if r_height == height else 150, r_height)


model_name = '{}-mix-{}{}-{}-H{}{}'.format(
    name_prefix,
    cnn_network.value,
    recurrent_network.value,
    size_str,
    hidden_num,
    name_suffix
)
trains_path = json.dumps(trains_path, ensure_ascii=False, indent=2).replace('\n', '\n  ')

BEST_LEARNING_RATE = {
    Optimizer.AdaBound: 0.001,
    Optimizer.Momentum: 0.01,
    Optimizer.Adam: 0.01,
    Optimizer.SGD: 0.01,
    Optimizer.RMSProp: 0.01,
    Optimizer.AdaGrad: 0.01,
}

learning_rate = BEST_LEARNING_RATE[optimizer] if not learning_rate else learning_rate


result = model.replace(
    "@trains_path", trains_path
).replace(
    "@model_name", model_name
).replace(
    "@resize", resize
).replace(
    "@size_str", size_str
).replace(
    "@width", str(width)
).replace(
    "@height", str(height)
).replace(
    "@charset", str(charset.value) if isinstance(charset, SimpleCharset) else str(charset)
).replace(
    "@test_num", str(test_num)
).replace(
    "@optimizer", str(optimizer.value)
).replace(
    "@hidden_num", str(hidden_num)
).replace(
    "@cnn_network", str(cnn_network.value)
).replace(
    "@recurrent_network", str(recurrent_network.value)
).replace(
    "@beam_width", str(beam_width)
).replace(
    "@learning_rate", str(learning_rate)
)
print(result)


with open("model.yaml".format(size_str), "w", encoding="utf8") as f:
    f.write(result)

from make_dataset import make_dataset
from trains import main
make_dataset()
main(None)