#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import io
import re
import cv2
import random
import PIL.Image
import numpy as np
import tensorflow as tf
from exception import *
from constants import RunMode
from config import ModelConfig, LabelFrom, LossFunction
from category import encode_maps
from pretreatment import preprocessing


class Encoder(object):
    """
    编码层：用于将数据输入编码为可输入网络的数据
    """
    def __init__(self, model_conf: ModelConfig, mode: RunMode):
        self.model_conf = model_conf
        self.mode = mode
        self.category_param = self.model_conf.category_param

    def image(self, path_or_bytes):
        """针对图片类型的输入的编码"""
        # im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # The OpenCV cannot handle gif format images, it will return None.
        # if im is None:
        path_or_stream = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes
        pil_image = PIL.Image.open(path_or_stream)
        rgb = pil_image.split()

        size = pil_image.size

        if len(rgb) > 3:
            background = PIL.Image.new('RGB', pil_image.size, (255, 255, 255))
            background.paste(pil_image, (0, 0, size[0], size[1]), pil_image)
            pil_image = background

        if self.model_conf.image_channel == 1:
            pil_image = pil_image.convert('L')

        im = np.array(pil_image)
        if self.mode == RunMode.Trains and bool(random.getrandbits(1)):
            im = preprocessing(
                image=im,
                binaryzation=self.model_conf.binaryzation,
                median_blur=self.model_conf.median_blur,
                gaussian_blur=self.model_conf.gaussian_blur,
                equalize_hist=self.model_conf.equalize_hist,
                laplacian=self.model_conf.laplace,
                rotate=self.model_conf.rotate,
                warp_perspective=self.model_conf.warp_perspective,
                sp_noise=self.model_conf.sp_noise,
            ).astype(np.float32)

        else:
            im = im.astype(np.float32)
        if self.model_conf.resize[0] == -1:
            # random_ratio = random.choice([2.5, 3, 3.5, 3.2, 2.7, 2.75])
            ratio = self.model_conf.resize[1] / size[1]
            # random_width = int(random_ratio * RESIZE[1])
            resize_width = int(ratio * size[0])
            # resize_width = random_width if is_random else resize_width
            im = cv2.resize(im, (resize_width, self.model_conf.resize[1]))
        else:
            im = cv2.resize(im, (self.model_conf.resize[0], self.model_conf.resize[1]))
        im = im.swapaxes(0, 1)

        if self.model_conf.image_channel == 1:
            return np.array((im[:, :, np.newaxis]) / 255.)
        else:
            return np.array(im[:, :]) / 255.

    def text(self, content, extracted=False):
        """针对文本类型的输入的编码"""
        if isinstance(content, bytes):
            content = content.decode("utf8")

        # 如果标签来源为文件名形如 aaa_md5.png
        if self.model_conf.label_from == LabelFrom.FileName:

            # 如果标签尚未提取解析
            if not extracted:
                found = re.search(self.model_conf.extract_regex, content)
                if not found:
                    exception(text="The regex is not extracted to the corresponding label", code=-777)
                found = found.group()
            else:
                found = content

            # 如果匹配内置的大小写规范，触发自动转换
            if isinstance(self.category_param, str) and '_LOWER' in self.category_param:
                found = found.lower()
            if isinstance(self.category_param, str) and '_UPPER' in self.category_param:
                found = found.upper()

            # 标签是否包含分隔符
            if self.model_conf.label_split:
                labels = found.split(self.model_conf.label_split)
            elif self.model_conf.max_label_num == 1:
                labels = [found]
            else:
                labels = [_ for _ in found]
            try:
                # 根据类别集合找到对应映射编码为dense数组
                if self.model_conf.loss_func == LossFunction.CTC:
                    label = self.split_continuous_char(
                        [encode_maps(self.model_conf.category)[i] for i in labels]
                    )
                else:
                    label = [encode_maps(self.model_conf.category)[i] for i in labels]
                return label

            except KeyError as e:
                exception(
                    'The sample label {} contains invalid charset: {}.'.format(
                        content, e.args[0]
                    ), ConfigException.SAMPLE_LABEL_ERROR
                )

    def split_continuous_char(self, content):
        # 为连续的分类插入空白符
        store_list = []
        for i in range(len(content) - 1):
            store_list.append(content[i])
            if content[i] == content[i + 1]:
                store_list += [self.model_conf.category_num]
        store_list.append(content[-1])
        return store_list


if __name__ == '__main__':
    pass



