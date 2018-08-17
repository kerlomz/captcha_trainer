#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import numpy as np
from utils import vec2text
from config import *


def predict_func(captcha_image, _sess, _predict, _x, _keep_prob):
    text_list = _sess.run(_predict, feed_dict={_x: [captcha_image], _keep_prob: 1})
    text = text_list[0].tolist()
    vector = np.zeros(MAX_CAPTCHA_LEN * CHAR_SET_LEN)
    i = 0
    for n in text:
        vector[i * CHAR_SET_LEN + n] = 1
        i += 1
    return vec2text(vector)