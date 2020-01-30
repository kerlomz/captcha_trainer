#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>

import os
import re
import base64
import pickle
from config import ModelConfig
from constants import ModelType
from config import COMPILE_MODEL_MAP


def parse_model(source_bytes: bytes, key=None):
    split_tag = b'-#||#-'

    if not key:
        key = [b"_____" + i.encode("utf8") + b"_____" for i in "&coriander"]
    if isinstance(key, str):
        key = [b"_____" + i.encode("utf8") + b"_____" for i in key]
    key_len_int = len(key)
    model_bytes_list = []
    graph_bytes_list = []
    slice_index = source_bytes.index(key[0])
    split_tag_len = len(split_tag)
    slice_0 = source_bytes[0: slice_index].split(split_tag)
    model_slice_len = len(slice_0[1])
    graph_slice_len = len(slice_0[0])
    slice_len = split_tag_len + model_slice_len + graph_slice_len

    for i in range(key_len_int-1):
        slice_index = source_bytes.index(key[i])
        print(slice_index, slice_index - slice_len)
        slices = source_bytes[slice_index - slice_len: slice_index].split(split_tag)
        model_bytes_list.append(slices[1])
        graph_bytes_list.append(slices[0])
    slices = source_bytes.split(key[-2])[1][:-len(key[-1])].split(split_tag)

    model_bytes_list.append(slices[1])
    graph_bytes_list.append(slices[0])
    model_bytes = b"".join(model_bytes_list)
    model_conf: ModelConfig = pickle.loads(model_bytes)
    graph_bytes: bytes = b"".join(graph_bytes_list)
    return model_conf, graph_bytes


def concat_model(output_path, model_bytes, graph_bytes, key=None):
    if not key:
        key = [b"_____" + i.encode("utf8") + b"_____" for i in "&coriander"]
    if isinstance(key, str):
        key = [b"_____" + i.encode("utf8") + b"_____" for i in key]
    key_len_int = len(key)
    model_slice_len = int(len(model_bytes) / key_len_int) + 1
    graph_slice_len = int(len(graph_bytes) / key_len_int) + 1
    model_slice = [model_bytes[i:i + model_slice_len] for i in range(0, len(model_bytes), model_slice_len)]

    graph_slice = [graph_bytes[i:i + graph_slice_len] for i in range(0, len(graph_bytes), graph_slice_len)]

    new_model = []
    for i in range(key_len_int):
        new_model.append(graph_slice[i] + b'-#||#-')
        new_model.append(model_slice[i])
        new_model.append(key[i])
    new_model = b"".join(new_model)
    with open(output_path, "wb") as f:
        f.write(new_model)
    print("Successfully write to model {}".format(output_path))


def output_model(project_name: str, model_type: ModelType, key=None):
    model_conf = ModelConfig(project_name, is_dev=False)

    graph_parent_path = model_conf.compile_model_path
    model_suffix = COMPILE_MODEL_MAP[model_type]
    model_bytes = pickle.dumps(model_conf.conf)
    graph_path = os.path.join(graph_parent_path, "{}{}".format(model_conf.model_name, model_suffix))

    with open(graph_path, "rb") as f:
        graph_bytes = f.read()

    output_path = graph_path.replace(".pb", ".pl").replace(".onnx", ".pl").replace(".tflite", ".pl")
    concat_model(output_path, model_bytes, graph_bytes, key)


if __name__ == '__main__':
    output_model("", ModelType.PB)

