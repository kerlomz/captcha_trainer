#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import cv2
import random
import PIL
import numpy as np
from math import floor, ceil


class Pretreatment(object):
    """
    预处理功能函数集合（目前仅用于训练过程中随机启动）
    """

    def __init__(self, origin):
        self.origin = origin

    def get(self):
        return self.origin

    def binarization(self, value: object, modify=False) -> np.ndarray:
        if isinstance(value, list) and len(value) == 2:
            value = random.randint(value[0], value[1])
        elif isinstance(value, int):
            value = value if (0 < value < 255) else -1
        if value == -1:
            return self.origin
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
        scale = 1.0
        height, width = size[0], size[1]
        center = (width // 2, height // 2)

        if bool(random.getrandbits(1)):
            angle = random.choice([
                -10, -20, -30, -45, -50, -60, -75, -90, -95, -100,
                10, 20, 30, 45, 50, 60, 75, 90, 95, 100
            ])
        else:
            angle = -random.randint(-value, value)

        m = cv2.getRotationMatrix2D(center, angle, scale)
        _rotate = cv2.warpAffine(self.origin, m, (width, height))
        # angle = -random.randint(-value, value)
        # if abs(angle) > 15:
        #     _img = cv2.resize(self.origin, (width, int(height / 2)))
        #     center = (width / 4, height / 4)
        # else:
        #     _img = cv2.resize(self.origin, (width, height))
        #     center = (width / 2, height / 2)
        # _img = cv2.resize(self.origin, (width, height))
        # m = cv2.getRotationMatrix2D(center, angle, 1.0)
        # _rotate = cv2.warpAffine(_img, m, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        if modify:
            self.origin = _rotate
        return _rotate

    # def warp_perspective(self, modify=False) -> np.ndarray:
    #     size = self.origin.shape
    #     height, width = size[0], size[1]
    #     size0 = random.randint(3, 9)
    #     size1 = random.randint(25, 30)
    #     size2 = random.randint(23, 27)
    #     size3 = random.randint(33, 37)
    #     pts1 = np.float32([[0, 0], [0, size1], [size1, size1], [size1, 0]])
    #     pts2 = np.float32([[size0, 0], [-size0, size1], [size2, size1], [size3, 0]])
    #     is_random = bool(random.getrandbits(1))
    #     param = (pts2, pts1) if is_random else (pts1, pts2)
    #     warp_mat = cv2.getPerspectiveTransform(*param)
    #     dst = cv2.warpPerspective(self.origin, warp_mat, (width, height))
    #     if modify:
    #         self.origin = dst
    #     return dst

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

    def random_brightness(self, modify=False):
        beta = np.random.uniform(-84, 84)
        output = np.uint8(np.clip((self.origin + beta), 0, 255))
        if modify:
            self.origin = output
        return output

    def random_saturation(self, modify=False):
        if len(self.origin.shape) < 3:
            return self.origin
        factor = np.random.uniform(0.3, 2.0)
        output = self.origin
        output[:, :, 1] = np.clip(output[:, :, 1] * factor, 0, 255)
        if modify:
            self.origin = output
        return output

    def random_hue(self, max_delta=18, modify=False):
        if len(self.origin.shape) < 3:
            return self.origin
        delta = np.random.uniform(-max_delta, max_delta)
        output = self.origin
        output[:, :, 0] = (output[:, :, 0] + delta) % 180.0
        if modify:
            self.origin = output
        return output

    def random_gamma(self, modify=False):
        if len(self.origin.shape) < 3:
            return self.origin
        gamma = np.random.uniform(0.25, 2.0)
        gamma_inv = 1.0 / gamma
        table = np.array([((i / 255.0) ** gamma_inv) * 255 for i in np.arange(0, 256)]).astype("uint8")
        output = cv2.LUT(self.origin, table)
        if modify:
            self.origin = output
        return output

    def random_channel_swap(self, modify=False):
        if len(self.origin.shape) < 3:
            return self.origin
        permutations = ((0, 2, 1),
                        (1, 0, 2), (1, 2, 0),
                        (2, 0, 1), (2, 1, 0))
        i = np.random.randint(5)
        order = permutations[i]
        output = self.origin[:, :, order]
        if modify:
            self.origin = output
        return output

    def random_blank(self, max_int, modify=False):
        if len(self.origin.shape) < 2:
            return self.origin
        corp_range_w = random.randint(0, max_int)
        corp_range_h = random.randint(0, max_int)
        output = self.origin
        random_p_h = -corp_range_h if bool(random.getrandbits(1)) else corp_range_h
        random_v_h = 255 if bool(random.getrandbits(1)) else 0
        random_p_w = -corp_range_w if bool(random.getrandbits(1)) else corp_range_w
        random_v_w = 255 if bool(random.getrandbits(1)) else 0
        if len(self.origin.shape) < 3:
            output[random_p_h, :] = 255 if bool(random.getrandbits(1)) else random_v_h
            output[:, random_p_w] = 255 if bool(random.getrandbits(1)) else random_v_w
        else:
            output[random_p_h, :, :] = 255 if bool(random.getrandbits(1)) else random_v_h
            output[:, random_p_w, :] = 255 if bool(random.getrandbits(1)) else random_v_w
        if modify:
            self.origin = output
        return output

    def random_transition(self, max_int, modify=False):
        size = self.origin.shape
        height, width = size[0], size[1]
        crop_range_w = random.randint(0, max_int)
        crop_range_w = crop_range_w if bool(random.getrandbits(1)) else -crop_range_w
        crop_range_h = random.randint(0, max_int)
        crop_range_h = crop_range_h if bool(random.getrandbits(1)) else -crop_range_h
        m = np.float32([[1, 0, crop_range_w], [0, 1, crop_range_h]])
        random_color = random.randint(240, 255)
        random_color = (random_color, random_color, random_color) if bool(random.getrandbits(1)) else (0, 0, 0)
        output = cv2.warpAffine(self.origin, m, (width, height), borderValue=random_color)
        if modify:
            self.origin = output
        return output

    def warp_perspective(self, modify=False):

        tmp = PIL.Image.fromarray(self.origin)
        w, h = tmp.size

        magnitude = random.randint(2, 4)
        grid_width = random.randint(5, 10)
        grid_height = random.randint(3, 7)

        horizontal_tiles = grid_width
        vertical_tiles = grid_height

        width_of_square = int(floor(w / float(horizontal_tiles)))
        height_of_square = int(floor(h / float(vertical_tiles)))

        width_of_last_square = w - (width_of_square * (horizontal_tiles - 1))
        height_of_last_square = h - (height_of_square * (vertical_tiles - 1))

        dimensions = []

        for vertical_tile in range(vertical_tiles):
            for horizontal_tile in range(horizontal_tiles):
                if vertical_tile == (vertical_tiles - 1) and horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_last_square + (horizontal_tile * width_of_square),
                                       height_of_last_square + (height_of_square * vertical_tile)])
                elif vertical_tile == (vertical_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_square + (horizontal_tile * width_of_square),
                                       height_of_last_square + (height_of_square * vertical_tile)])
                elif horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_last_square + (horizontal_tile * width_of_square),
                                       height_of_square + (height_of_square * vertical_tile)])
                else:
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_square + (horizontal_tile * width_of_square),
                                       height_of_square + (height_of_square * vertical_tile)])

        # For loop that generates polygons could be rewritten, but maybe harder to read?
        # polygons = [x1,y1, x1,y2, x2,y2, x2,y1 for x1,y1, x2,y2 in dimensions]

        # last_column = [(horizontal_tiles - 1) + horizontal_tiles * i for i in range(vertical_tiles)]
        last_column = []
        for i in range(vertical_tiles):
            last_column.append((horizontal_tiles - 1) + horizontal_tiles * i)

        last_row = range((horizontal_tiles * vertical_tiles) - horizontal_tiles, horizontal_tiles * vertical_tiles)

        polygons = []
        for x1, y1, x2, y2 in dimensions:
            polygons.append([x1, y1, x1, y2, x2, y2, x2, y1])

        polygon_indices = []
        for i in range((vertical_tiles * horizontal_tiles) - 1):
            if i not in last_row and i not in last_column:
                polygon_indices.append([i, i + 1, i + horizontal_tiles, i + 1 + horizontal_tiles])

        for a, b, c, d in polygon_indices:
            dx = random.randint(-magnitude, magnitude)
            dy = random.randint(-magnitude, magnitude)

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[a]
            polygons[a] = [x1, y1,
                           x2, y2,
                           x3 + dx, y3 + dy,
                           x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[b]
            polygons[b] = [x1, y1,
                           x2 + dx, y2 + dy,
                           x3, y3,
                           x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[c]
            polygons[c] = [x1, y1,
                           x2, y2,
                           x3, y3,
                           x4 + dx, y4 + dy]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[d]
            polygons[d] = [x1 + dx, y1 + dy,
                           x2, y2,
                           x3, y3,
                           x4, y4]

        generated_mesh = []
        for i in range(len(dimensions)):
            generated_mesh.append([dimensions[i], polygons[i]])

        output = tmp.transform(tmp.size, PIL.Image.MESH, generated_mesh, resample=PIL.Image.BICUBIC)
        if modify:
            self.origin = np.asarray(output)
        return output


def preprocessing(
        image,
        binaryzation=-1,
        median_blur=-1,
        gaussian_blur=-1,
        equalize_hist=False,
        laplacian=False,
        warp_perspective=False,
        sp_noise=-1.0,
        rotate=-1,
        random_blank=-1,
        random_transition=-1,
        random_brightness=False,
        random_gamma=False,
        random_channel_swap=False,
        random_saturation=False,
        random_hue=False,
):
    """
    各种预处理函数是否启用及参数配置
    :param random_transition: bool, 随机位移
    :param random_blank: bool, 随机填充空白
    :param random_brightness: bool, 随机亮度
    :param image: numpy图片数组,
    :param binaryzation: list-int数字范围, 二值化
    :param median_blur: int数字,
    :param gaussian_blur: int数字,
    :param equalize_hist: bool,
    :param laplacian: bool, 拉普拉斯
    :param warp_perspective: bool, 透视变形
    :param sp_noise: 浮点, 椒盐噪声
    :param rotate: 数字, 旋转
    :param corp: 裁剪
    :return:
    """
    pretreatment = Pretreatment(image)
    if rotate > 0:
        pretreatment.rotate(rotate, True)
    if random_transition != -1:
        pretreatment.random_transition(5, True)
    if 0 < sp_noise < 1:
        pretreatment.sp_noise(sp_noise, True)
    if binaryzation != -1:
        pretreatment.binarization(binaryzation, True)
    if median_blur != -1:
        pretreatment.median_blur(median_blur, True)
    if gaussian_blur != -1:
        pretreatment.gaussian_blur(gaussian_blur, True)
    if equalize_hist and bool(random.getrandbits(1)):
        pretreatment.equalize_hist(True, True)
    if laplacian:
        pretreatment.laplacian(True, True)
    if warp_perspective:
        pretreatment.warp_perspective(True)
    if random_brightness and bool(random.getrandbits(1)):
        pretreatment.random_brightness(True)
    if random_blank != -1:
        pretreatment.random_blank(2, True)
    if random_gamma and bool(random.getrandbits(1)):
        pretreatment.random_gamma(True)
    if random_channel_swap and bool(random.getrandbits(1)):
        pretreatment.random_channel_swap(True)
    if random_saturation and bool(random.getrandbits(1)):
        pretreatment.random_saturation(True)
    if random_hue and bool(random.getrandbits(1)):
        pretreatment.random_hue(18, True)
    return pretreatment.get()


def preprocessing_by_func(exec_map: dict, src_arr, key=None):
    if not exec_map:
        return src_arr
    target_arr = cv2.cvtColor(src_arr, cv2.COLOR_RGB2BGR)
    if not key:
        key = random.choice(list(exec_map.keys()))
    for sentence in exec_map.get(key):
        if sentence.startswith("@@"):
            target_arr = eval(sentence[2:])
        elif sentence.startswith("$$"):
            exec(sentence[2:])
    return cv2.cvtColor(target_arr, cv2.COLOR_BGR2RGB)


if __name__ == '__main__':
    import io
    import os
    import PIL.Image
    import random

    root_dir = r"H:\TrainSet\九宫格\极验\切图"
    name = random.choice(os.listdir(root_dir))
    # name = "3956_b8cee4da-3530-11ea-9778-c2f9192435fa.png"
    path = os.path.join(root_dir, name)
    print(path)
    with open(path, "rb") as f:
        path_or_bytes = f.read()
    path_or_stream = io.BytesIO(path_or_bytes)
    pil_image = PIL.Image.open(path_or_stream).convert("RGB")
    im = np.array(pil_image)
    # im = preprocessing_by_func(exec_map={
    #     "black": [
    #         "$$target_arr[:, :, 2] = 255 - target_arr[:, :, 2]",
    #     ],
    #     "red": [],
    #     "yellow": [
    #         "$$target_arr[:, :, 2] = 255 - target_arr[:, :, 2]",
    #         "@@target_arr[:, :, (0, 2, 0)]",
    #         "$$target_arr[:, :, 2] = 255 - target_arr[:, :, 2]",
    #
    #         # "$$target_arr[:, :, 2] = 255 - target_arr[:, :, 2]",
    #         # "@@target_arr[:, :, (0, 2, 1)]",
    #
    #         # "$$target_arr[:, :, 1] = 255 - target_arr[:, :, 1]",
    #         # "@@target_arr[:, :, (2, 1, 0)]",
    #         # "@@target_arr[:, :, (1, 2, 0)]",
    #     ],
    #     "blue": [
    #         "@@target_arr[:, :, (1, 2, 0)]",
    #     ]
    # },
    #     src_arr=im,
    #     key="blue"
    # )
    im = preprocessing(
        image=im,
        # binaryzation=150,
        # warp_perspective=True,
        random_transition=True,
        # rotate=90
    ).astype(np.float32)
    # im = im.swapaxes(0, 1)
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    cv_img = cv2.imencode('.png', im)[1]
    img_bytes = bytes(bytearray(cv_img))
    with open(r"1.jpg", "wb") as f:
        f.write(img_bytes)
