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

    def binarization(self, value, modify=False) -> np.ndarray:
        ret, _binarization = cv2.threshold(self.origin, value, 255, cv2.THRESH_BINARY)
        if modify:
            self.origin = _binarization
        return _binarization

    def median_blur(self, value, modify=False) -> np.ndarray:
        if not value:
            return self.origin
        value = random.randint(0, value)
        value = value + 1 if value % 2 == 0 else value
        _smooth = cv2.medianBlur(self.origin, value)
        if modify:
            self.origin = _smooth
        return _smooth

    def gaussian_blur(self, value, modify=False) -> np.ndarray:
        if not value:
            return self.origin
        value = random.randint(0, value)
        value = value + 1 if value % 2 == 0 else value
        _blur = cv2.GaussianBlur(self.origin, (value, value), 0)
        if modify:
            self.origin = _blur
        return _blur

    def equalize_hist(self, value, modify=False) -> np.ndarray:
        if not value:
            return self.origin
        _equalize_hist = cv2.equalizeHist(self.origin)
        if modify:
            self.origin = _equalize_hist
        return _equalize_hist

    def laplacian(self, value, modify=False) -> np.ndarray:
        if not value:
            return self.origin
        _laplacian = cv2.convertScaleAbs(cv2.Laplacian(self.origin, cv2.CV_16S, ksize=3))
        if modify:
            self.origin = _laplacian
        return _laplacian

    def rotate(self, value, modify=False):
        if not value:
            return self.origin
        size = self.origin.shape
        height, width = size[0], size[1]
        angle = -random.randint(-value, value)
        if abs(angle) > 15:
            _img = cv2.resize(self.origin, (width, int(height / 2)))
            center = (width / 4, height / 4)
        else:
            _img = cv2.resize(self.origin, (width, height))
            center = (width / 2, height / 2)
        _img = cv2.resize(self.origin, (width, height))
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        _rotate = cv2.warpAffine(_img, m, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        if modify:
            self.origin = _rotate
        return _rotate


def preprocessing(image, binaryzation=-1, median_blur=-1, gaussian_blur=-1, equalize_hist=False, laplacian=False, rotate=-1):
    pretreatment = Pretreatment(image)
    if binaryzation > 0 and bool(random.getrandbits(1)):
        pretreatment.binarization(binaryzation, True)
    if median_blur != -1 and bool(random.getrandbits(1)):
        pretreatment.median_blur(median_blur, True)
    if gaussian_blur != -1 and bool(random.getrandbits(1)):
        pretreatment.gaussian_blur(gaussian_blur, True)
    if equalize_hist and bool(random.getrandbits(1)):
        pretreatment.equalize_hist(True, True)
    if laplacian and bool(random.getrandbits(1)):
        pretreatment.laplacian(True, True)
    if rotate > 0 and bool(random.getrandbits(1)):
        pretreatment.rotate(rotate, True)
    return pretreatment.get()


if __name__ == '__main__':
    pass
