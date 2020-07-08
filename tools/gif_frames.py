#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>

import cv2
import numpy as np
from PIL import ImageSequence


def split_frames(image_obj, need_frame=None):
    if not need_frame:
        need_frame = [0]
    image_seq = ImageSequence.all_frames(image_obj)
    image_arr_last = [np.asarray(image_seq[-1])] if -1 in need_frame and len(need_frame) > 1 else []
    image_arr = [np.asarray(item) for i, item in enumerate(image_seq) if (i in need_frame or need_frame == [-1])]
    image_arr += image_arr_last
    return image_arr


def concat_arr(img_arr):
    if len(img_arr) < 1:
        return img_arr
    all_slice = img_arr[0]
    for im_slice in img_arr[1:]:
        all_slice = np.concatenate((all_slice, im_slice), axis=1)
    return all_slice


def numpy_to_bytes(numpy_arr):
    cv_img = cv2.imencode('.png', numpy_arr)[1]
    img_bytes = bytes(bytearray(cv_img))
    return img_bytes


def concat_frames(image_obj, need_frame=None):
    if not need_frame:
        need_frame = [0]
    img_arr = split_frames(image_obj, need_frame)
    img_arr = concat_arr(img_arr)
    return img_arr


def blend_arr(img_arr):
    if len(img_arr) < 1:
        return img_arr
    all_slice = img_arr[0]
    for im_slice in img_arr[1:]:
        all_slice = cv2.addWeighted(all_slice, 0.5, im_slice, 0.5, 0)
    # print(all_slice)
    # all_slice = cv2.equalizeHist(all_slice)
    return all_slice


def blend_frame(image_obj, need_frame=None):
    if not need_frame:
        need_frame = [-1]
    img_arr = split_frames(image_obj, need_frame)
    img_arr = blend_arr(img_arr)
    if len(img_arr.shape) > 2 and img_arr.shape[2] == 3:
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    img_arr = cv2.equalizeHist(img_arr)
    return img_arr


if __name__ == "__main__":
    pass