#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import cv2


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


if __name__ == '__main__':
    pass
