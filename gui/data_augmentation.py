#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import json
import tkinter as tk
import tkinter.ttk as ttk
from gui.utils import LayoutGUI


class DataAugmentationDialog(tk.Toplevel):

    def __init__(self):
        tk.Toplevel.__init__(self)
        self.title('Data Augmentation')
        self.layout = {
            'global': {
                'start': {'x': 15, 'y': 20},
                'space': {'x': 15, 'y': 25},
                'tiny_space': {'x': 5, 'y': 10}
            }
        }
        self.data_augmentation_entity = None
        self.da_random_captcha = {"Enable": False, "FontPath": ""}
        self.window_width = 750
        self.window_height = 220

        self.layout_utils = LayoutGUI(self.layout, self.window_width)
        screenwidth = self.winfo_screenwidth()
        screenheight = self.winfo_screenheight()
        size = '%dx%d+%d+%d' % (
            self.window_width,
            self.window_height,
            (screenwidth - self.window_width) / 2,
            (screenheight - self.window_height) / 2
        )
        self.geometry(size)
        # ============================= Group 4 =====================================
        self.label_frame_augmentation = ttk.Labelframe(self, text='Data Augmentation')
        self.label_frame_augmentation.place(
            x=self.layout['global']['start']['x'],
            y=self.layout['global']['start']['y'],
            width=725,
            height=150
        )

        # 二值化 - 标签
        self.binaryzation_text = ttk.Label(self, text='Binaryzation', anchor=tk.W)
        self.layout_utils.inside_widget(
            src=self.binaryzation_text,
            target=self.label_frame_augmentation,
            width=72,
            height=20,
        )

        # 二值化 - 输入框
        self.binaryzation_val = tk.StringVar()
        self.binaryzation_val.set(-1)
        self.binaryzation_entry = ttk.Entry(self, textvariable=self.binaryzation_val, justify=tk.LEFT)
        self.layout_utils.next_to_widget(
            src=self.binaryzation_entry,
            target=self.binaryzation_text,
            width=55,
            height=20,
            tiny_space=True
        )

        # 滤波 - 标签
        self.median_blur_text = ttk.Label(self, text='Median Blur', anchor=tk.W)
        self.layout_utils.next_to_widget(
            src=self.median_blur_text,
            target=self.binaryzation_entry,
            width=80,
            height=20,
            tiny_space=False
        )

        # 滤波 - 输入框
        self.median_blur_val = tk.IntVar()
        self.median_blur_val.set(-1)
        self.median_blur_entry = ttk.Entry(self, textvariable=self.median_blur_val, justify=tk.LEFT)
        self.layout_utils.next_to_widget(
            src=self.median_blur_entry,
            target=self.median_blur_text,
            width=52,
            height=20,
            tiny_space=True
        )

        # 高斯模糊 - 标签
        self.gaussian_blur_text = ttk.Label(self, text='Gaussian Blur', anchor=tk.W)
        self.layout_utils.next_to_widget(
            src=self.gaussian_blur_text,
            target=self.median_blur_entry,
            width=85,
            height=20,
            tiny_space=False
        )

        # 高斯模糊 - 输入框
        self.gaussian_blur_val = tk.IntVar()
        self.gaussian_blur_val.set(-1)
        self.gaussian_blur_entry = ttk.Entry(self, textvariable=self.gaussian_blur_val, justify=tk.LEFT)
        self.layout_utils.next_to_widget(
            src=self.gaussian_blur_entry,
            target=self.gaussian_blur_text,
            width=62,
            height=20,
            tiny_space=True
        )

        # 椒盐噪声 - 标签
        self.sp_noise_text = ttk.Label(self, text='Pepper Noise (0-1)', anchor=tk.W)
        self.layout_utils.next_to_widget(
            src=self.sp_noise_text,
            target=self.gaussian_blur_entry,
            width=110,
            height=20,
            tiny_space=False
        )

        # 椒盐噪声 - 输入框
        self.sp_noise_val = tk.DoubleVar()
        self.sp_noise_val.set(-1)
        self.sp_noise_entry = ttk.Entry(self, textvariable=self.sp_noise_val, justify=tk.LEFT)
        self.layout_utils.next_to_widget(
            src=self.sp_noise_entry,
            target=self.sp_noise_text,
            width=71,
            height=20,
            tiny_space=True
        )

        # 旋转 - 标签
        self.rotate_text = ttk.Label(self, text='Rotate (0-90)', anchor=tk.W)
        self.layout_utils.below_widget(
            src=self.rotate_text,
            target=self.binaryzation_text,
            width=72,
            height=20,
            tiny_space=True
        )

        # 旋转 - 输入框
        self.rotate_val = tk.IntVar()
        self.rotate_val.set(-1)
        self.rotate_entry = ttk.Entry(self, textvariable=self.rotate_val, justify=tk.LEFT)
        self.layout_utils.next_to_widget(
            src=self.rotate_entry,
            target=self.rotate_text,
            width=55,
            height=20,
            tiny_space=True
        )

        # 随机空白边缘 - 标签
        self.random_blank_text = ttk.Label(self, text='Blank Border', anchor=tk.W)
        self.layout_utils.next_to_widget(
            src=self.random_blank_text,
            target=self.rotate_entry,
            width=72,
            height=20,
            tiny_space=False
        )

        # 随机空白边缘 - 输入框
        self.random_blank_val = tk.IntVar()
        self.random_blank_val.set(-1)
        self.random_blank_entry = ttk.Entry(self, textvariable=self.random_blank_val, justify=tk.LEFT)
        self.layout_utils.next_to_widget(
            src=self.random_blank_entry,
            target=self.random_blank_text,
            width=55,
            height=20,
            tiny_space=True
        )

        # 随机边缘位移 - 标签
        self.random_transition_text = ttk.Label(self, text='Transition', anchor=tk.W)
        self.layout_utils.next_to_widget(
            src=self.random_transition_text,
            target=self.random_blank_entry,
            width=60,
            height=20,
            tiny_space=False
        )

        # 随机边缘位移 - 输入框
        self.random_transition_val = tk.IntVar()
        self.random_transition_val.set(-1)
        self.random_transition_entry = ttk.Entry(self, textvariable=self.random_transition_val, justify=tk.LEFT)
        self.layout_utils.next_to_widget(
            src=self.random_transition_entry,
            target=self.random_transition_text,
            width=55,
            height=20,
            tiny_space=True
        )

        # 随机验证码字体 - 标签
        self.random_captcha_font_text = ttk.Label(self, text='RandomCaptcha - Font', anchor=tk.W)
        self.layout_utils.next_to_widget(
            src=self.random_captcha_font_text,
            target=self.random_transition_entry,
            width=130,
            height=20,
            tiny_space=False
        )

        # 随机验证码字体
        self.random_captcha_font_val = tk.StringVar()
        self.random_captcha_font_val.set("")
        self.random_captcha_font_entry = ttk.Entry(self, textvariable=self.random_captcha_font_val, justify=tk.LEFT)
        self.layout_utils.next_to_widget(
            src=self.random_captcha_font_entry,
            target=self.random_captcha_font_text,
            width=75,
            height=20,
            tiny_space=True
        )

        # 透视变换 - 多选框
        self.warp_perspective_val = tk.IntVar()
        self.warp_perspective_val.set(0)
        self.warp_perspective = ttk.Checkbutton(
            self, text='Distortion', variable=self.warp_perspective_val, onvalue=1, offvalue=0
        )
        self.layout_utils.below_widget(
            src=self.warp_perspective,
            target=self.rotate_text,
            width=80,
            height=20,
            tiny_space=False
        )

        # 直方图均衡化 - 多选框
        self.equalize_hist_val = tk.IntVar()
        self.equalize_hist_val.set(0)
        self.equalize_hist = ttk.Checkbutton(
            self, text='EqualizeHist', variable=self.equalize_hist_val, offvalue=0
        )
        self.layout_utils.next_to_widget(
            src=self.equalize_hist,
            target=self.warp_perspective,
            width=100,
            height=20,
            tiny_space=True
        )

        # 拉普拉斯 - 多选框
        self.laplace_val = tk.IntVar()
        self.laplace_val.set(0)
        self.laplace = ttk.Checkbutton(
            self, text='Laplace', variable=self.laplace_val, onvalue=1, offvalue=0
        )
        self.layout_utils.next_to_widget(
            src=self.laplace,
            target=self.equalize_hist,
            width=64,
            height=20,
            tiny_space=True
        )

        # 随机亮度 - 多选框
        self.brightness_val = tk.IntVar()
        self.brightness_val.set(0)
        self.brightness = ttk.Checkbutton(
            self, text='Brightness', variable=self.brightness_val, offvalue=0
        )
        self.layout_utils.next_to_widget(
            src=self.brightness,
            target=self.laplace,
            width=80,
            height=20,
            tiny_space=True
        )

        # 随机饱和度 - 多选框
        self.saturation_val = tk.IntVar()
        self.saturation_val.set(0)
        self.saturation = ttk.Checkbutton(
            self, text='Saturation', variable=self.saturation_val, offvalue=0
        )
        self.layout_utils.next_to_widget(
            src=self.saturation,
            target=self.brightness,
            width=80,
            height=20,
            tiny_space=True
        )

        # 随机色相 - 多选框
        self.hue_val = tk.IntVar()
        self.hue_val.set(0)
        self.hue = ttk.Checkbutton(
            self, text='Hue', variable=self.hue_val, offvalue=0
        )
        self.layout_utils.next_to_widget(
            src=self.hue,
            target=self.saturation,
            width=50,
            height=20,
            tiny_space=True
        )

        # 随机Gamma - 多选框
        self.gamma_val = tk.IntVar()
        self.gamma_val.set(0)
        self.gamma = ttk.Checkbutton(
            self, text='Gamma', variable=self.gamma_val, offvalue=0
        )
        self.layout_utils.next_to_widget(
            src=self.gamma,
            target=self.hue,
            width=80,
            height=20,
            tiny_space=True
        )

        # 随机通道 - 多选框
        self.channel_swap_val = tk.IntVar()
        self.channel_swap_val.set(0)
        self.channel_swap = ttk.Checkbutton(
            self, text='Channel Swap', variable=self.channel_swap_val, offvalue=0
        )
        self.layout_utils.next_to_widget(
            src=self.channel_swap,
            target=self.gamma,
            width=100,
            height=20,
            tiny_space=True
        )

        # 保存 - 按钮
        self.btn_save = ttk.Button(self, text='Save Configuration', command=lambda: self.save_conf())
        self.layout_utils.widget_from_right(
            src=self.btn_save,
            target=self.label_frame_augmentation,
            width=120,
            height=24,
            tiny_space=True
        )

    def read_conf(self, entity):
        self.data_augmentation_entity = entity
        self.binaryzation_val.set(json.dumps(entity.binaryzation))
        self.median_blur_val.set(entity.median_blur)
        self.gaussian_blur_val.set(entity.gaussian_blur)
        self.equalize_hist_val.set(entity.equalize_hist)
        self.laplace_val.set(entity.laplace)
        self.warp_perspective_val.set(entity.warp_perspective)
        self.rotate_val.set(entity.rotate)
        self.sp_noise_val.set(entity.sp_noise)
        self.brightness_val.set(entity.brightness)
        self.saturation_val.set(entity.saturation)
        self.hue_val.set(entity.hue)
        self.gamma_val.set(entity.gamma)
        self.channel_swap_val.set(entity.channel_swap)
        self.random_blank_val.set(entity.random_blank)
        self.random_transition_val.set(entity.random_transition)
        self.da_random_captcha = entity.random_captcha
        self.random_captcha_font_val.set(self.da_random_captcha['FontPath'])

    def save_conf(self):
        self.data_augmentation_entity.binaryzation = json.loads(self.binaryzation_val.get()) if self.binaryzation_val else []
        self.data_augmentation_entity.median_blur = self.median_blur_val.get()
        self.data_augmentation_entity.gaussian_blur = self.gaussian_blur_val.get()
        self.data_augmentation_entity.rotate = self.rotate_val.get()
        self.data_augmentation_entity.sp_noise = self.sp_noise_val.get()
        self.data_augmentation_entity.random_blank = self.random_blank_val.get()
        self.data_augmentation_entity.random_transition = self.random_transition_val.get()

        if self.random_captcha_font_val.get():
            self.data_augmentation_entity.random_captcha['Enable'] = True
            self.data_augmentation_entity.random_captcha['FontPath'] = self.random_captcha_font_val.get()
        else:
            self.data_augmentation_entity.random_captcha['Enable'] = False
            self.data_augmentation_entity.random_captcha['FontPath'] = ""

        self.data_augmentation_entity.equalize_hist = True if self.equalize_hist_val.get() == 1 else False
        self.data_augmentation_entity.laplace = True if self.laplace_val.get() == 1 else False
        self.data_augmentation_entity.warp_perspective = True if self.warp_perspective_val.get() == 1 else False

        self.data_augmentation_entity.brightness = True if self.brightness_val.get() == 1 else False
        self.data_augmentation_entity.saturation = True if self.saturation_val.get() == 1 else False
        self.data_augmentation_entity.hue = True if self.hue_val.get() == 1 else False
        self.data_augmentation_entity.gamma = True if self.gamma_val.get() == 1 else False
        self.data_augmentation_entity.channel_swap = True if self.channel_swap_val.get() == 1 else False

        self.destroy()