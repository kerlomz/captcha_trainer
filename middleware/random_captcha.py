from PIL import Image, ImageDraw, ImageFont
from enum import Enum, unique
from fontTools.ttLib import TTFont
import numpy as np
import io
import os
import base64
import hashlib
import time
import random
import logging


class BackgroundType(Enum):
    RANDOM = 'random'
    IMAGE = 'image'
    RGB = 'rgb'


class RandomCaptcha(object):
    def __init__(self):
        self.__width = [130, 160]
        self.__height = [50, 60]
        self.__background_mode = BackgroundType.RGB
        self.__background_img_assests_path = None
        self.__rgb = {
            'r': [0, 255],
            'g': [0, 255],
            'b': [0, 255]
        }
        self.__fonts_list = []
        self.__samples = []
        self.__fonts_num = [4, 4]
        self.__font_size = [26, 36]
        self.__font_mode = 0
        self.__max_line_count = 2
        self.__max_point_count = 20

    @property
    def max_point_count(self):
        return self.__max_point_count

    @max_point_count.setter
    def max_point_count(self, value: int):
        self.__max_point_count = value

    @property
    def max_line_count(self):
        return self.__max_line_count

    @max_line_count.setter
    def max_line_count(self, value: int):
        self.__max_line_count = value

    @property
    def font_mode(self):
        return self.__font_mode

    @font_mode.setter
    def font_mode(self, value: int):
        self.__font_mode = value

    @property
    def font_size(self) -> list:
        return self.__font_size

    @font_size.setter
    def font_size(self, value: list):
        if type(value) == list and type(value[0]) == int and type(value[1]) == int and value[0] >= 0 and value[1] > 0 and value[0] < value[1]:
            self.__font_size = value
        else:
            raise ValueError("input value should be like [0, 255]")

    @property
    def fonts_num(self) -> list:
        return self.__fonts_num

    @fonts_num.setter
    def fonts_num(self, value: list):
        self.__fonts_num = value

    @property
    def sample(self) -> list:
        return self.__samples

    @sample.setter
    def sample(self, value: list):
        self.__samples = value

    @property
    def fonts_list(self) -> list:
        return self.__fonts_list

    @fonts_list.setter
    def fonts_list(self, value: list):
        self.__fonts_list = value

    @property
    def rgb(self) -> dict:
        return self.__rgb

    @property
    def rgb_r(self) -> list:
        return self.__rgb['r']

    @rgb_r.setter
    def rgb_r(self, value: list):
        if type(value) == list and type(value[0]) == int and type(value[1]) == int and value[0] >= 0 and value[1] > 0 and value[0] < value[1] and value[0] <= 255:
            self.__rgb['r'] = value
        else:
            raise ValueError("input value should be like [0, 255]")

    @property
    def rgb_g(self) -> list:
        return self.__rgb['g']

    @rgb_g.setter
    def rgb_g(self, value: list):
        if type(value) == list and type(value[0]) == int and type(value[1]) == int and value[0] >= 0 and value[1] > 0 and value[0] < value[1] and value[0] <= 255:
            self.__rgb['g'] = value
        else:
            raise ValueError("input value should be like [0, 255]")

    @property
    def rgb_b(self) -> list:
        return self.__rgb['b']

    @rgb_b.setter
    def rgb_b(self, value: list):
        if type(value) == list and type(value[0]) == int and type(value[1]) == int and value[0] >= 0 and value[1] > 0 and value[0] < value[1]:
            self.__rgb['b'] = value
        else:
            raise ValueError("input value should be like [0, 255]")

    @property
    def background_mode(self) -> BackgroundType:
        return self.__background_mode

    @background_mode.setter
    def background_mode(self, value: BackgroundType):
        self.__background_mode = value

    @property
    def background_img_path(self) -> str:
        return self.__background_img_assests_path

    @background_img_path.setter
    def background_img_path(self, value: str):
        self.__background_img_assests_path = value

    @property
    def height(self):
        return self.__height

    @height.setter
    def height(self, value):
        self.__height = value

    @property
    def width(self):
        return self.__width

    @width.setter
    def width(self, value):
        self.__width = value

    def check_font(self):
        for font_type in self.fonts_list:
            try:
                font = TTFont(font_type)
                uni_map = font['cmap'].tables[0].ttFont.getBestCmap()
                for item in self.sample:
                    codepoint = ord(str(item))
                    if codepoint in uni_map.keys():
                        continue
                    else:
                        font.close()
                        raise Exception("{} not found!".format(item))
            except Exception as e:
                try:
                    os.remove(font_type)
                except:
                    pass
                del self.fonts_list[self.fonts_list.index(font_type)]

        pass

    def set_text(self, __image: ImageDraw, img_width, img_height):

        if img_width >= 150:
            font_size = random.choice(range(self.font_size[0], self.font_size[1]))
        else:
            font_size = random.choice(range(self.font_size[0], int((self.font_size[0] + self.font_size[1])/2)))

        font_num = random.choice(range(self.fonts_num[0], self.fonts_num[1]))
        max_width = int(img_width / font_num)
        max_height = int(img_height)
        font_type = random.choice(self.fonts_list)
        try:
            font = ImageFont.truetype(font_type, font_size)
        except OSError:
            del self.fonts_list[self.fonts_list.index(font_type)]
            raise Exception("{} opened fail")
        labels = []
        for idx in range(font_num):
            fw = range(int(max_width - font_size))
            if len(fw) > 0:
                x = max_width * idx + random.choice(fw)
            else:
                x = max_width * idx
            y = random.choice(range(int(max_height - font_size)))
            f = random.choice(self.sample)
            labels.append(f)
            __image.text((x, y), f, font=font,
                         fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        return labels, font_type

    def set_noise(self, __image: ImageDraw, img_width, img_height):
        for i in range(self.max_line_count):
            # 噪线的起点横坐标和纵坐标
            x1 = random.randint(0, img_width)
            y1 = random.randint(0, img_height)
            # 噪线的终点横坐标和纵坐标
            x2 = random.randint(0, img_width)
            y2 = random.randint(0, img_height)
            # 通过画笔对象draw.line((起点的xy, 终点的xy), fill='颜色')来划线
            __image.line((x1, y1, x2, y2),
                         fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        for i in range(self.max_point_count):
            __image.point([random.randint(0, img_width), random.randint(0, img_height)], fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
            x = random.randint(0, img_width)
            y = random.randint(0, img_height)
            __image.arc((x, y, x + 4, y + 4), 0, 40, fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

    def set_content(self, __image: ImageDraw, img_width, img_height):
        labels, font_type = self.set_text(__image, img_width, img_height)
        self.set_noise(__image, img_width, img_height)
        return labels, font_type

    def create(self, mode: str = "bytes", img_format: str = "png"):
        if type(self.width) == list:
            img_width = random.choice(range(self.width[0], self.width[1]))
        else:
            img_width = self.width
        if type(self.height) == list:
            img_height = random.choice(range(self.height[0], self.height[1]))
        else:
            img_height = self.height

        background_mode = self.background_mode
        if type(background_mode) is BackgroundType:
            if background_mode.value == BackgroundType.RGB.value:
                rgb_range = self.rgb
                r_range = rgb_range['r']
                g_range = rgb_range['g']
                b_range = rgb_range['b']
                rgb = (random.randint(r_range[0], r_range[1]), random.randint(g_range[0], g_range[1]),
                       random.randint(b_range[0], b_range[1]))
                __image = Image.new('RGB', (img_width, img_height), rgb)
                img = ImageDraw.Draw(__image)
                labels, font_type = self.set_content(img, img_width, img_height)
                if mode == "bytes":
                    img_byte_arr = io.BytesIO()
                    __image.save(img_byte_arr, format=img_format)
                    return img_byte_arr.getvalue(), labels, font_type
                elif mode == "numpy":
                    return np.array(__image), labels, font_type
                elif mode == "base64":
                    img_byte_arr = io.BytesIO()
                    __image.save(img_byte_arr, format=img_format)
                    _bytes = img_byte_arr.getvalue()
                    return base64.b64encode(_bytes).decode(), labels, font_type
                else:
                    raise FutureWarning("暂不支持的输出类型")
            else:
                raise FutureWarning("暂不支持的背景类型")
        else:
            raise TypeError("background mode must be BGMODEL.")
