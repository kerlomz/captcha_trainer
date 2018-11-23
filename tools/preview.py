#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import random
import numpy as np
from tkinter import *
from PIL import Image as Pil_Image, ImageTk
from PIL import ImageFile
from config import *
from utils import preprocessing

ImageFile.LOAD_TRUNCATED_IMAGES = True


def path2list(path, shuffle=False):
    _path = path[0] if isinstance(path, list) else path
    file_list = os.listdir(_path)
    group = [os.path.join(_path, image_file) for image_file in file_list if not image_file.startswith(".")]
    if shuffle:
        random.shuffle(group)
    return group


TRAINS_GROUP = path2list(TRAINS_PATH, True)


def fetch():
    image_path = TRAINS_GROUP[random.randint(0, len(TRAINS_GROUP))]
    image = Pil_Image.open(image_path).convert('L')
    captcha_image = preprocessing(np.array(image), BINARYZATION, SMOOTH, BLUR)
    captcha_image = Pil_Image.fromarray(captcha_image)
    captcha_image = captcha_image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    return image_path, ImageTk.PhotoImage(captcha_image)


def set_captcha():
    path, image = fetch()
    print(path)
    captcha.config(image=image)
    captcha.image = image
    label_path['text'] = path
    root.update()


def get_screen_size(window):
    return window.winfo_screenwidth(), window.winfo_screenheight()


def get_window_size(window):
    return window.winfo_reqwidth(), window.winfo_reqheight()


def center_window(width, height):
    screenwidth = root.winfo_screenwidth()
    screenheight = root.winfo_screenheight()
    size = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
    root.geometry(size)


if __name__ == '__main__':
    root = Tk()
    root.title('Pretreatment Previewer')
    center_window(500, 150)
    captcha = Label(root)
    captcha.bind('<Button-1>', lambda _: set_captcha())
    label_path = Label(root, text='', font=('Consolas', 10))
    label_path.place(x=5, y=5)
    captcha.place(x=5, y=30)
    set_captcha()
    root.mainloop()


