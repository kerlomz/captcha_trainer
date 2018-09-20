#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import random
from tkinter import *
from PIL import Image as Pil_Image, ImageTk
from PIL import ImageFile
from config import *
from utils import path2list, preprocessing


ImageFile.LOAD_TRUNCATED_IMAGES = True
TRAINS_GROUP = path2list(TRAINS_PATH, True)


def fetch():
    image_path = TRAINS_GROUP[random.randint(0, len(TRAINS_GROUP))]
    image = Pil_Image.open(image_path)
    captcha_image = preprocessing(image, BINARYZATION, SMOOTH, BLUR, IMAGE_ORIGINAL_COLOR, INVERT)
    captcha_image = Pil_Image.fromarray(captcha_image)
    captcha_image = captcha_image.resize(RESIZE if RESIZE else (IMAGE_WIDTH, IMAGE_HEIGHT))
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


