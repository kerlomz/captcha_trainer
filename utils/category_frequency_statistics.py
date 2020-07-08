#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import re
import os
import json
from config import ModelConfig, LabelFrom, DatasetType

ignore_list = ["Thumbs.db", ".DS_Store"]
PATH_SPLIT = "/"


def extract_labels_from_filename(filename: str, extract_regex):
    if filename.split("/")[-1] in ignore_list:
        return None
    try:
        labels = re.search(extract_regex, filename.split(PATH_SPLIT)[-1])
    except re.error as e:
        print('error:', e)
        return None
    if labels:
        labels = labels.group()
    else:
        print('invalid filename {}, ignored.'.format(filename))
        return None
    return labels


def fetch_category_freq(model: ModelConfig):
    if model.label_from == LabelFrom.FileName:
        category_dict = dict()
        for iter_dir in model.trains_path[DatasetType.Directory]:
            for filename in os.listdir(iter_dir):

                labels = extract_labels_from_filename(filename, model.extract_regex)

                if not labels:
                    continue

                for label_item in labels:
                    if label_item in category_dict:
                        category_dict[label_item] += 1
                    else:
                        category_dict[label_item] = 0

        return sorted(category_dict.items(), key=lambda item: item[1], reverse=True)


def fetch_category_list(model: ModelConfig, is_json=False):
    if model.label_from == LabelFrom.FileName:
        category_set = set()
        for iter_dir in model.trains_path[DatasetType.Directory]:
            for filename in os.listdir(iter_dir):

                labels = extract_labels_from_filename(filename, model.extract_regex)

                if not labels:
                    continue

                for label_item in labels:
                    category_set.add(label_item)
        category_list = list(category_set)
        category_list.sort()
        if is_json:
            return json.dumps(category_list, ensure_ascii=False)
        return category_list


if __name__ == '__main__':
    model_conf = ModelConfig("test-CNNX-GRU-H64-CTC-C1")
    # labels_dict = fetch_category_freq(model_conf)
    # label_list = [k for k, v in labels_dict if v < 5000]
    # label_list.sort()
    # high_freq = "".join(label_list)
    # print(high_freq)
    # print(len(high_freq))
    labels_list = fetch_category_list(model_conf)
    print(labels_list)