#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import os
import numpy as np
import cv2
import random
from config import CHAR_SET_LEN, GEN_CHAR_SET, MAX_CAPTCHA_LEN


def char2pos(c):
    return GEN_CHAR_SET.index(c)


def pos2char(char_idx):
    return GEN_CHAR_SET[char_idx]


def vec2text(vec):
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_idx = c % CHAR_SET_LEN
        char_code = pos2char(char_idx)
        text.append(char_code)
    return "".join(text)


def text2vec(text):
    text_len = len(text)
    if text_len > MAX_CAPTCHA_LEN:
        raise ValueError('Sample label {} exceeds the maximum length of the defined captcha label. \n'
                         'Please match the value corresponding to CharLength of model.yaml'.format(text))
    vector = np.zeros(MAX_CAPTCHA_LEN * CHAR_SET_LEN)
    try:
        for i, c in enumerate(text):
            idx = i * CHAR_SET_LEN + char2pos(c)
            vector[idx] = 1
    except ValueError:
        print("ValueError", text)
    return vector


def path2list(path, shuffle=False):

    file_list = os.listdir(path)
    group = [os.path.join(path, image_file) for image_file in file_list if not image_file.startswith(".")]
    if shuffle:
        random.shuffle(group)
    return group


def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        return gray
    else:
        return img


def preprocessing(pil_image, binaryzation=127, smooth=-1, blur=-1, original_color=False, invert=False):
    _pil_image = pil_image
    if not original_color:
        _pil_image = _pil_image.convert("L")
    image = np.array(_pil_image)

    # if not original_color:
    #     image = convert2gray(image)
    if binaryzation > 0:
        ret, thresh = cv2.threshold(image, binaryzation, 255, cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY)
    else:
        thresh = image
    _image = thresh
    if smooth != -1:
        smooth = smooth + 1 if smooth % 2 == 0 else smooth
        _smooth = cv2.medianBlur(thresh, smooth)
        _image = _smooth
    if blur != -1:
        blur = blur + 1 if blur % 2 == 0 else blur
        _blur = cv2.GaussianBlur(_image if smooth != -1 else thresh, (blur, blur), 0)
        _image = _blur
    return _image
