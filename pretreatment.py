#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import cv2
import random
import numpy as np


class Pretreatment(object):

    def __init__(self, origin):
        self.origin = origin

    def get(self):
        return self.origin

    def binarization(self, value, modify=False):
        ret, _binarization = cv2.threshold(self.origin, value, 255, cv2.THRESH_BINARY)
        if modify:
            self.origin = _binarization
        return _binarization

    def median_blur(self, value, modify=False):
        if not value:
            return self.origin
        value = value + 1 if value % 2 == 0 else value
        _smooth = cv2.medianBlur(self.origin, value)
        if modify:
            self.origin = _smooth
        return _smooth

    def gaussian_blur(self, value, modify=False):
        if not value:
            return self.origin
        value = value + 1 if value % 2 == 0 else value
        _blur = cv2.GaussianBlur(self.origin, (value, value), 0)
        if modify:
            self.origin = _blur
        return _blur


def preprocessing(image, binaryzation=-1, smooth=-1, blur=-1):
    pretreatment = Pretreatment(image)
    if binaryzation > 0:
        pretreatment.binarization(binaryzation, True)
    if smooth != -1:
        pretreatment.median_blur(smooth, True)
    if blur != -1:
        pretreatment.gaussian_blur(blur, True)
    return pretreatment.get()


def equalize_hist(img) -> np.ndarray:
    return cv2.equalizeHist(img)


def laplacian(img) -> np.ndarray:
    return cv2.convertScaleAbs(cv2.Laplacian(img, cv2.CV_16S, ksize=3))


def warp_perspective(img) -> np.ndarray:
    size = img.shape
    height, width = size[0], size[1]
    size0 = random.randint(3, 9)
    size1 = random.randint(25, 30)
    size2 = random.randint(23, 27)
    size3 = random.randint(33, 37)
    pts1 = np.float32([[0, 0], [0, size1], [size1, size1], [size1, 0]])
    pts2 = np.float32([[size0, 0], [-size0, size1], [size2, size1], [size3, 0]])
    is_random = bool(random.getrandbits(1))
    param = (pts2, pts1) if is_random else (pts1, pts2)
    warp_mat = cv2.getPerspectiveTransform(*param)
    dst = cv2.warpPerspective(img, warp_mat, (width, height))
    return dst


def rotate(img):
    size = img.shape
    height, width = size[0], size[1]
    angle = -random.randint(-4, 4)
    if abs(angle) > 15:
        _img = cv2.resize(img, (width, int(height / 2)))
        center = (width / 4, height / 4)
    else:
        _img = cv2.resize(img, (width, height))
        center = (width / 2, height / 2)
    _img = cv2.resize(img, (width, height))
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(_img, m, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


if __name__ == '__main__':
    pass
