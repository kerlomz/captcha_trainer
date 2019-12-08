#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import cv2
import random
import numpy as np


class Pretreatment(object):
    """
    预处理功能函数集合（目前仅用于训练过程中随机启动）
    """
    def __init__(self, origin):
        self.origin = origin

    def get(self):
        return self.origin

    def binarization(self, value, modify=False) -> np.ndarray:
        if isinstance(value, list) and len(value) == 2:
            value = random.randint(value[0], value[1])
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

    def rotate(self, value, modify=False) -> np.ndarray:
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

    def warp_perspective(self, modify=False) -> np.ndarray:
        size = self.origin.shape
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
        dst = cv2.warpPerspective(self.origin, warp_mat, (width, height))
        if modify:
            self.origin = dst
        return dst

    def sp_noise(self, prob, modify=False):
        size = self.origin.shape
        output = np.zeros(self.origin.shape, np.uint8)
        thres = 1 - prob
        for i in range(size[0]):
            for j in range(size[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = self.origin[i][j]
        if modify:
            self.origin = output
        return output

    def light(self, modify=False):
        alpha = 0.3
        beta = random.randint(0, 80)
        alpha = alpha * 0.01
        output = np.uint8(np.clip((alpha * self.origin + beta), 0, 255))
        if modify:
            self.origin = output
        return output


def preprocessing(
        image,
        binaryzation=-1,
        median_blur=-1,
        gaussian_blur=-1,
        equalize_hist=False,
        laplacian=False,
        warp_perspective=False,
        sp_noise=-1,
        rotate=-1,
        light=False
):
    """
    各种预处理函数是否启用及参数配置
    :param light: bool
    :param image: numpy图片数组
    :param binaryzation: list-int数字范围
    :param median_blur: int数字
    :param gaussian_blur: int数字
    :param equalize_hist: bool
    :param laplacian: bool
    :param warp_perspective: bool
    :param sp_noise: 浮点
    :param rotate: 数字
    :return:
    """
    pretreatment = Pretreatment(image)
    if binaryzation != -1 and bool(random.getrandbits(1)):
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
    if warp_perspective and bool(random.getrandbits(1)):
        pretreatment.warp_perspective(True)
    if 0 < sp_noise < 1 and bool(random.getrandbits(1)):
        pretreatment.sp_noise(sp_noise, True)
    if light and bool(random.getrandbits(1)):
        pretreatment.light(True)
    return pretreatment.get()


if __name__ == '__main__':
    pass
