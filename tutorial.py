#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import os
import re
import json
import PIL.Image as pilImage
from constants import *
from config import PATH_SPLIT

# - [ALPHANUMERIC, ALPHANUMERIC_LOWER, ALPHANUMERIC_UPPER,
# -- NUMERIC, ALPHABET_LOWER, ALPHABET_UPPER, ALPHABET, ALPHANUMERIC_LOWER_MIX_CHINESE_3500]

category = SimpleCharset.ALPHANUMERIC

cnn_network = CNNNetwork.CNN5
recurrent_network = RecurrentNetwork.GRU
optimizer = Optimizer.AdaBound
loss = LossFunction.CTC

trains_path = [
    r"D:\*",
]

validation_set_num = 50
hidden_num = 16
learning_rate = None

name_prefix = 'name_prefix'
name_suffix = None
name_prefix = name_prefix if name_prefix else "tutorial"
name_suffix = '-' + str(name_suffix) if name_suffix else ''

# trains_path = [i.replace("\\", "/") for i in trains_path]
file_name = os.listdir(trains_path[0])[0]
default_regex = '.*?(?=_)'
size = pilImage.open(os.path.join(trains_path[0], file_name)).size
label_sample = re.search(default_regex, file_name.split(PATH_SPLIT)[-1]).group()
label_num = len(label_sample)
width = size[0]
height = size[1]


size_str = "{}x{}".format(width, height)

resize = "[{}, {}]".format(width, height)
validation_batch = validation_set_num if validation_set_num < 300 else 300
model_name = '{}-mix-{}{}-{}-H{}-{}{}'.format(
    name_prefix,
    cnn_network.value,
    recurrent_network.value,
    size_str,
    hidden_num,
    loss.value,
    name_suffix
)

project_path = "./projects/{}".format(model_name)
if not os.path.exists(project_path):
    os.makedirs(project_path)

model_conf_path = os.path.join(project_path, "model.yaml")

trains_path = "".join(["\n    - " + i for i in trains_path])

BEST_LEARNING_RATE = {
    Optimizer.AdaBound: 0.001,
    Optimizer.Momentum: 0.01,
    Optimizer.Adam: 0.01,
    Optimizer.SGD: 0.01,
    Optimizer.RMSProp: 0.01,
    Optimizer.AdaGrad: 0.01,
}

dataset_trains_name = "dataset/{}_trains.tfrecords".format(model_name)
dataset_validation_name = "dataset/{}_validation.tfrecords".format(model_name)
dataset_trains_path = os.path.join(project_path, dataset_trains_name)
dataset_validation_path = os.path.join(project_path, dataset_validation_name)
learning_rate = BEST_LEARNING_RATE[optimizer] if not learning_rate else learning_rate


with open("model.template", encoding="utf8") as f:
    base_config = "".join(f.readlines())
    model = base_config.format(
        MemoryUsage=0.7,
        CNNNetwork=cnn_network.value,
        RecurrentNetwork=recurrent_network.value,
        HiddenNum=hidden_num,
        Optimizer=optimizer.value,
        LossFunction=loss.value,
        Decoder=loss.value,
        ModelName=model_name,
        ModelField=ModelField.Image.value,
        ModelScene=ModelScene.Classification.value,
        Category=category.value,
        Resize=resize,
        ImageChannel=1,
        ImageWidth=width,
        ImageHeight=height,
        MaxLabelNum=label_num if loss == LossFunction.CrossEntropy else -1,
        LabelFrom=LabelFrom.FileName.value,
        ExtractRegex='.*?(?=_)',
        Split='null',
        TrainsPath=dataset_trains_path,
        ValidationPath=dataset_validation_path,
        DatasetPath=trains_path,
        ValidationSetNum=validation_set_num,
        SavedSteps=100,
        ValidationSteps=500,
        EndAcc=0.95,
        EndCost=0.1,
        EndEpochs=2,
        BatchSize=64,
        ValidationBatchSize=validation_batch,
        LearningRate=learning_rate,
        Binaryzation=-1,
        MedianBlur=-1,
        GaussianBlur=-1,
        EqualizeHist=-1,
        Laplace=False,
        Rotate=False
    )


with open(model_conf_path, "w", encoding="utf8") as f:
    f.write(model)

from make_dataset import DataSets
from trains import main
from config import ModelConfig
model = ModelConfig(model_name)
DataSets(model).make_dataset()
main([model_name])
