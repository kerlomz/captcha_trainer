#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
from PIL import Image
from PIL import ImageFile
from config import *
from utils import preprocessing, path2list

ImageFile.LOAD_TRUNCATED_IMAGES = True
TRAINS_GROUP = path2list(TRAINS_PATH, True)
image_path = TRAINS_GROUP[random.randint(0, len(TRAINS_GROUP))]
image = Image.open(image_path)
captcha = preprocessing(image, BINARYZATION, SMOOTH, BLUR, IMAGE_ORIGINAL_COLOR, INVERT)
captcha = Image.fromarray(captcha)
captcha = captcha.resize(RESIZE if RESIZE else (IMAGE_WIDTH, IMAGE_HEIGHT))
print(image_path)
captcha.show()

