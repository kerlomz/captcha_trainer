#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import io
import re
import cv2
import PIL.Image
import numpy as np
from exception import *
from constants import RunMode
from config import ModelConfig, LabelFrom, LossFunction
from category import encode_maps
from pretreatment import preprocessing


class Encoder(object):
    def __init__(self, model_conf: ModelConfig, mode: RunMode):
        self.model_conf = model_conf
        self.mode = mode

    def image(self, path_or_bytes):

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
        if self.mode == RunMode.Trains:
            im = preprocessing(
                image=im,
                binaryzation=self.model_conf.binaryzation,
                median_blur=self.model_conf.median_blur,
                gaussian_blur=self.model_conf.gaussian_blur,
                equalize_hist=self.model_conf.equalize_hist,
                laplacian=self.model_conf.laplace,
                rotate=self.model_conf.rotate
            ).astype(np.float32)

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
        if isinstance(content, bytes):
            content = content.decode("utf8")
        if self.model_conf.label_from == LabelFrom.FileName:
            if not extracted:
                found = re.search(self.model_conf.extract_regex, content)
                if not found:
                    exception(text="The regex is not extracted to the corresponding label", code=-777)
                found = found.group()
            else:
                found = content
            if self.model_conf.label_split:
                labels = found.split(self.model_conf.label_split)
            else:
                labels = [_ for _ in found]
            try:
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
        store_list = []
        for i in range(len(content) - 1):
            store_list.append(content[i])
            if content[i] == content[i + 1]:
                store_list += [self.model_conf.category_num]
        store_list.append(content[-1])
        return store_list


if __name__ == '__main__':
    _m = ModelConfig("demo")
    a = Encoder(_m).text("v898999_yuiyui.png")
    print(a)



