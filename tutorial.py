#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import os
import json
import PIL.Image as pil_image


class RecurrentNetwork:
    LSTM = 'LSTM'
    BLSTM = 'BLSTM'
    SRU = 'SRU'
    BSRU = 'BSRU'


charset = "ALPHANUMERIC_LOWER"
network = RecurrentNetwork.BLSTM
trains_path = [
    r"D:\TrainSet\***",
]

model = """
# ModelName: Corresponding to the model file in the model directory,
# - such as YourModelName.pb, fill in YourModelName here.
# CharSet: Provides a default optional built-in solution:
# - [ALPHANUMERIC, ALPHANUMERIC_LOWER, ALPHANUMERIC_UPPER,
# -- NUMERIC, ALPHABET_LOWER, ALPHABET_UPPER, ALPHABET, ALPHANUMERIC_LOWER_MIX_CHINESE_3500]
# - Or you can use your own customized character set like: ['a', '1', '2'].
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
Pretreatment:
  Binaryzation: -1
  Smoothing: -1
  Blur: -1
  Resize: @resize

Trains:
#  TrainsPath: './dataset/@model_name_trains.tfrecords'
#  TestPath: './dataset/@model_name_test.tfrecords'
  TrainsPath: @trains_path
  
"""

# - [ALPHANUMERIC, ALPHANUMERIC_LOWER, ALPHANUMERIC_UPPER,
# -- NUMERIC, ALPHABET_LOWER, ALPHABET_UPPER, ALPHABET, ALPHANUMERIC_LOWER_MIX_CHINESE_3500]

trains_path = [i.replace("\\", "/") for i in trains_path]
file_name = os.listdir(trains_path[0])[0]
size = pil_image.open(os.path.join(trains_path[0], file_name)).size

width = size[0]
height = size[1]
size_str = "{}x{}".format(width, height)
if width > 180 or width < 120:
    r_height = int(height * 150 / width)
else:
    r_height = height
resize = "[{}, {}]".format(width if r_height == height else 150, r_height)

model_name = 'sell-mix-CNN5{}-{}'.format(network, size_str)
trains_path = json.dumps(trains_path, ensure_ascii=False).replace("]", "  ]")
result = model.replace("@trains_path", trains_path).replace("@model_name", model_name).replace("@resize", resize).replace("@size_str", size_str).replace("@width", str(width)).replace("@height", str(height)).replace("@charset", charset)
print(result)

with open("model.yaml".format(size_str), "w", encoding="utf8") as f:
    f.write(result)

from make_dataset import run
from trains import main
run()
with open("model1.yaml".format(size_str), "w") as f:
    f.write("\n".join(result.split("\n")[:-3]).replace("#  TrainsPath", "  TrainsPath").replace("#  TestPath", "  TestPath"))
main(None)