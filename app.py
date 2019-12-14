import os
import re
import math
import sys
import shutil
import json
import traceback
import numpy.core._dtype_ctypes
import PIL.Image as PilImage
import threading
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from tkinter import filedialog
from constants import *
from config import ModelConfig, OUTPUT_SHAPE1_MAP, NETWORK_MAP
from make_dataset import DataSets
from trains import Trains
from category import category_extract, SIMPLE_CATEGORY_MODEL


class Wizard:

    job: threading.Thread
    current_task: Trains
    is_task_running: bool = False

    def __init__(self, parent):
        self.layout = {
            'global': {
                'start': {'x': 15, 'y': 20},
                'space': {'x': 15, 'y': 25},
                'tiny_space': {'x': 5, 'y': 10}
            }
        }
        self.parent = parent
        self.parent.iconbitmap(Wizard.resource_path("resource/icon.ico"))
        self.current_project: str = ""
        self.project_root_path = "./projects"
        if not os.path.exists(self.project_root_path):
            os.makedirs(self.project_root_path)
        self.parent.title('Image Classification Wizard Tool based on Deep Learning')
        self.parent.resizable(width=False, height=False)
        self.window_width = 815
        self.window_height = 780
        screenwidth = self.parent.winfo_screenwidth()
        screenheight = self.parent.winfo_screenheight()
        size = '%dx%d+%d+%d' % (
            self.window_width,
            self.window_height,
            (screenwidth - self.window_width) / 2,
            (screenheight - self.window_height) / 2
        )
        self.parent.geometry(size)

        self.parent.bind('<Button-1>', lambda x: self.blank_click(x))
        # ============================= Group 1 =====================================
        self.label_frame_source = ttk.Labelframe(self.parent, text='Sample Source')
        self.label_frame_source.place(
            x=self.layout['global']['start']['x'],
            y=self.layout['global']['start']['y'],
            width=790,
            height=150
        )

        # 训练集源路径 - 标签
        self.dataset_train_path_text = ttk.Label(self.parent, text='Training Path', anchor=tk.W)
        self.inside_widget(
            src=self.dataset_train_path_text,
            target=self.label_frame_source,
            width=90,
            height=20
        )

        # 训练集源路径 - 输入控件
        self.source_train_path_listbox = tk.Listbox(self.parent, font=('微软雅黑', 9))
        self.next_to_widget(
            src=self.source_train_path_listbox,
            target=self.dataset_train_path_text,
            width=600,
            height=50,
            tiny_space=True
        )
        self.source_train_path_listbox.bind(
            sequence="<Delete>",
            func=lambda x: self.listbox_delete_item_callback(x, self.source_train_path_listbox)
        )
        self.listbox_scrollbar(self.source_train_path_listbox)

        # 训练集源路径 - 按钮
        self.btn_browse_train = ttk.Button(
            self.parent, text='Browse', command=lambda: self.browse_dataset(DatasetType.Directory, RunMode.Trains)
        )
        self.next_to_widget(
            src=self.btn_browse_train,
            target=self.source_train_path_listbox,
            width=60,
            height=24,
            tiny_space=True
        )

        # 验证集源路径 - 标签
        label_edge = self.object_edge_info(self.dataset_train_path_text)
        widget_edge = self.object_edge_info(self.source_train_path_listbox)
        self.dataset_validation_path_text = ttk.Label(self.parent, text='Validation Path', anchor=tk.W)
        self.dataset_validation_path_text.place(
            x=label_edge['x'],
            y=widget_edge['edge_y'] + self.layout['global']['space']['y'] / 2,
            width=90,
            height=20
        )

        # 验证集源路径 - 输入控件
        self.source_validation_path_listbox = tk.Listbox(self.parent, font=('微软雅黑', 9))
        self.next_to_widget(
            src=self.source_validation_path_listbox,
            target=self.dataset_validation_path_text,
            width=600,
            height=50,
            tiny_space=True
        )
        self.source_validation_path_listbox.bind(
            sequence="<Delete>",
            func=lambda x: self.listbox_delete_item_callback(x, self.source_validation_path_listbox)
        )
        self.listbox_scrollbar(self.source_validation_path_listbox)

        # 训练集源路径 - 按钮
        self.btn_browse_validation = ttk.Button(
            self.parent, text='Browse', command=lambda: self.browse_dataset(DatasetType.Directory, RunMode.Validation)
        )
        self.next_to_widget(
            src=self.btn_browse_validation,
            target=self.source_validation_path_listbox,
            width=60,
            height=24,
            tiny_space=True
        )

        # ============================= Group 2 =====================================
        self.label_frame_neu = ttk.Labelframe(self.parent, text='Neural network')
        self.below_widget(
            src=self.label_frame_neu,
            target=self.label_frame_source,
            width=790,
            height=120,
            tiny_space=False
        )

        # 最大标签数目 - 标签
        self.label_num_text = ttk.Label(self.parent, text='Label Num', anchor=tk.W)
        self.inside_widget(
            src=self.label_num_text,
            target=self.label_frame_neu,
            width=65,
            height=20,
        )

        # 最大标签数目 - 滚动框
        self.label_num_spin = ttk.Spinbox(self.parent, from_=1, to=12)
        self.label_num_spin.set(1)
        self.next_to_widget(
            src=self.label_num_spin,
            target=self.label_num_text,
            width=50,
            height=20,
            tiny_space=True
        )

        # 图像通道 - 标签
        self.channel_text = ttk.Label(self.parent, text='Channel', anchor=tk.W)
        self.next_to_widget(
            src=self.channel_text,
            target=self.label_num_spin,
            width=50,
            height=20,
            tiny_space=False
        )

        # 图像通道 - 下拉框
        self.comb_channel = ttk.Combobox(self.parent, values=(3, 1), state='readonly')
        self.comb_channel.current(0)
        self.next_to_widget(
            src=self.comb_channel,
            target=self.channel_text,
            width=38,
            height=20,
            tiny_space=True
        )

        # 卷积层 - 标签
        self.neu_cnn_text = ttk.Label(self.parent, text='CNN Layer', anchor=tk.W)
        self.next_to_widget(
            src=self.neu_cnn_text,
            target=self.comb_channel,
            width=65,
            height=20,
            tiny_space=False
        )

        # 卷积层 - 下拉框
        self.comb_neu_cnn = ttk.Combobox(self.parent, values=[_.name for _ in CNNNetwork], state='readonly')
        self.comb_neu_cnn.current(0)
        self.next_to_widget(
            src=self.comb_neu_cnn,
            target=self.neu_cnn_text,
            width=80,
            height=20,
            tiny_space=True
        )

        # 循环层 - 标签
        self.neu_recurrent_text = ttk.Label(self.parent, text='Recurrent Layer', anchor=tk.W)
        self.next_to_widget(
            src=self.neu_recurrent_text,
            target=self.comb_neu_cnn,
            width=95,
            height=20,
            tiny_space=False
        )

        # 循环层 - 下拉框
        self.comb_recurrent = ttk.Combobox(self.parent, values=[_.name for _ in RecurrentNetwork], state='readonly')
        self.comb_recurrent.current(0)
        self.next_to_widget(
            src=self.comb_recurrent,
            target=self.neu_recurrent_text,
            width=112,
            height=20,
            tiny_space=True
        )
        self.comb_recurrent.bind("<<ComboboxSelected>>", lambda x: self.auto_loss(x))

        # 循环层单元数 - 标签
        self.units_num_text = ttk.Label(self.parent, text='UnitsNum', anchor=tk.W)
        self.next_to_widget(
            src=self.units_num_text,
            target=self.comb_recurrent,
            width=60,
            height=20,
            tiny_space=False
        )

        # 循环层单元数 - 下拉框
        self.units_num_spin = ttk.Spinbox(self.parent, from_=16, to=512, increment=16, wrap=True)
        self.units_num_spin.set(64)
        self.next_to_widget(
            src=self.units_num_spin,
            target=self.units_num_text,
            width=55,
            height=20,
            tiny_space=True
        )

        # 损失函数 - 标签
        self.loss_func_text = ttk.Label(self.parent, text='Loss Function', anchor=tk.W)
        self.below_widget(
            src=self.loss_func_text,
            target=self.label_num_text,
            width=85,
            height=20,
            tiny_space=True
        )

        # 损失函数 - 下拉框
        self.comb_loss = ttk.Combobox(self.parent, values=[_.name for _ in LossFunction], state='readonly')
        self.comb_loss.current(0)
        self.next_to_widget(
            src=self.comb_loss,
            target=self.loss_func_text,
            width=101,
            height=20,
            tiny_space=True
        )

        # 优化器 - 标签
        self.optimizer_text = ttk.Label(self.parent, text='Optimizer', anchor=tk.W)
        self.next_to_widget(
            src=self.optimizer_text,
            target=self.comb_loss,
            width=60,
            height=20,
            tiny_space=False
        )

        # 优化器 - 下拉框
        self.comb_optimizer = ttk.Combobox(self.parent, values=[_.name for _ in Optimizer], state='readonly')
        self.comb_optimizer.current(0)
        self.next_to_widget(
            src=self.comb_optimizer,
            target=self.optimizer_text,
            width=88,
            height=20,
            tiny_space=True
        )

        # 学习率 - 标签
        self.learning_rate_text = ttk.Label(self.parent, text='Learning Rate', anchor=tk.W)
        self.next_to_widget(
            src=self.learning_rate_text,
            target=self.comb_optimizer,
            width=85,
            height=20,
            tiny_space=False
        )

        # 学习率 - 滚动框
        self.learning_rate_spin = ttk.Spinbox(self.parent, from_=0.00001, to=0.1, increment='0.0001')
        self.learning_rate_spin.set(0.001)
        self.next_to_widget(
            src=self.learning_rate_spin,
            target=self.learning_rate_text,
            width=67,
            height=20,
            tiny_space=True
        )

        # Resize - 标签
        self.resize_text = ttk.Label(self.parent, text='Resize', anchor=tk.W)
        self.next_to_widget(
            src=self.resize_text,
            target=self.learning_rate_spin,
            width=36,
            height=20,
            tiny_space=False
        )

        # Resize - 输入框
        self.resize_val = tk.StringVar()
        self.resize_val.set('[150, 50]')
        self.resize_entry = ttk.Entry(self.parent, textvariable=self.resize_val, justify=tk.LEFT)
        self.next_to_widget(
            src=self.resize_entry,
            target=self.resize_text,
            width=60,
            height=20,
            tiny_space=True
        )

        # Size - 标签
        self.size_text = ttk.Label(self.parent, text='Size', anchor=tk.W)
        self.next_to_widget(
            src=self.size_text,
            target=self.resize_entry,
            width=30,
            height=20,
            tiny_space=False
        )

        # Size - 输入框
        self.size_val = tk.StringVar()
        self.size_val.set('[-1, -1]')
        self.size_entry = ttk.Entry(self.parent, textvariable=self.size_val, justify=tk.LEFT)
        self.next_to_widget(
            src=self.size_entry,
            target=self.size_text,
            width=60,
            height=20,
            tiny_space=True
        )

        # 类别 - 标签
        self.category_text = ttk.Label(self.parent, text='Category', anchor=tk.W)
        self.below_widget(
            src=self.category_text,
            target=self.loss_func_text,
            width=72,
            height=20,
            tiny_space=True
        )

        # 类别 - 下拉框
        self.comb_category = ttk.Combobox(self.parent, values=(
            'CUSTOMIZED',
            'NUMERIC',
            'ALPHANUMERIC',
            'ALPHANUMERIC_LOWER',
            'ALPHANUMERIC_UPPER',
            'ALPHABET_LOWER',
            'ALPHABET_UPPER',
            'ALPHABET',
            'ARITHMETIC',
            'FLOAT',
            'CHS_3500',
            'ALPHANUMERIC_CHS_3500_LOWER'
        ), state='readonly')
        self.comb_category.current(1)
        self.comb_category.bind("<<ComboboxSelected>>", lambda x: self.comb_category_callback(x))
        self.next_to_widget(
            src=self.comb_category,
            target=self.category_text,
            width=225,
            height=20,
            tiny_space=True
        )

        # 类别 - 自定义输入框
        self.category_val = tk.StringVar()
        self.category_val.set('')
        self.category_entry = ttk.Entry(self.parent, textvariable=self.category_val, justify=tk.LEFT, state=tk.DISABLED)
        self.next_to_widget(
            src=self.category_entry,
            target=self.comb_category,
            width=440,
            height=20,
            tiny_space=False
        )

        # ============================= Group 3 =====================================
        self.label_frame_train = ttk.Labelframe(self.parent, text='Training Configuration')
        self.below_widget(
            src=self.label_frame_train,
            target=self.label_frame_neu,
            width=790,
            height=60,
            tiny_space=True
        )

        # 任务完成标准 - 准确率 - 标签
        self.end_acc_text = ttk.Label(self.parent, text='End Accuracy', anchor=tk.W)
        self.inside_widget(
            src=self.end_acc_text,
            target=self.label_frame_train,
            width=85,
            height=20,
        )

        # 任务完成标准 - 准确率 - 输入框
        self.end_acc_val = tk.DoubleVar()
        self.end_acc_val.set(0.95)
        self.end_acc_entry = ttk.Entry(self.parent, textvariable=self.end_acc_val, justify=tk.LEFT)
        self.next_to_widget(
            src=self.end_acc_entry,
            target=self.end_acc_text,
            width=56,
            height=20,
            tiny_space=True
        )

        # 任务完成标准 - 平均损失 - 标签
        self.end_cost_text = ttk.Label(self.parent, text='End Cost', anchor=tk.W)
        self.next_to_widget(
            src=self.end_cost_text,
            target=self.end_acc_entry,
            width=60,
            height=20,
            tiny_space=False
        )

        # 任务完成标准 - 平均损失 - 输入框
        self.end_cost_val = tk.DoubleVar()
        self.end_cost_val.set(0.5)
        self.end_cost_entry = ttk.Entry(self.parent, textvariable=self.end_cost_val, justify=tk.LEFT)
        self.next_to_widget(
            src=self.end_cost_entry,
            target=self.end_cost_text,
            width=58,
            height=20,
            tiny_space=True
        )

        # 任务完成标准 - 循环轮次 - 标签
        self.end_epochs_text = ttk.Label(self.parent, text='End Epochs', anchor=tk.W)
        self.next_to_widget(
            src=self.end_epochs_text,
            target=self.end_cost_entry,
            width=72,
            height=20,
            tiny_space=False
        )

        # 任务完成标准 - 循环轮次 - 输入框
        self.end_epochs_spin = ttk.Spinbox(self.parent, from_=0, to=10000)
        self.end_epochs_spin.set(2)
        self.next_to_widget(
            src=self.end_epochs_spin,
            target=self.end_epochs_text,
            width=50,
            height=20,
            tiny_space=True
        )

        # 训练批次大小 - 标签
        self.batch_size_text = ttk.Label(self.parent, text='Train BatchSize', anchor=tk.W)
        self.next_to_widget(
            src=self.batch_size_text,
            target=self.end_epochs_spin,
            width=90,
            height=20,
            tiny_space=False
        )

        # 训练批次大小 - 输入框
        self.batch_size_val = tk.IntVar()
        self.batch_size_val.set(64)
        self.batch_size_entry = ttk.Entry(self.parent, textvariable=self.batch_size_val, justify=tk.LEFT)
        self.next_to_widget(
            src=self.batch_size_entry,
            target=self.batch_size_text,
            width=40,
            height=20,
            tiny_space=True
        )

        # 验证批次大小 - 标签
        self.validation_batch_size_text = ttk.Label(self.parent, text='Validation BatchSize', anchor=tk.W)
        self.next_to_widget(
            src=self.validation_batch_size_text,
            target=self.batch_size_entry,
            width=120,
            height=20,
            tiny_space=False
        )

        # 验证批次大小 - 输入框
        self.validation_batch_size_val = tk.IntVar()
        self.validation_batch_size_val.set(300)
        self.validation_batch_size_entry = ttk.Entry(self.parent, textvariable=self.validation_batch_size_val, justify=tk.LEFT)
        self.next_to_widget(
            src=self.validation_batch_size_entry,
            target=self.validation_batch_size_text,
            width=40,
            height=20,
            tiny_space=True
        )

        # ============================= Group 4 =====================================
        self.label_frame_augmentation = ttk.Labelframe(self.parent, text='Data Augmentation')
        self.below_widget(
            src=self.label_frame_augmentation,
            target=self.label_frame_train,
            width=790,
            height=90,
            tiny_space=True
        )

        # 二值化 - 标签
        self.binaryzation_text = ttk.Label(self.parent, text='Binaryzation', anchor=tk.W)
        self.inside_widget(
            src=self.binaryzation_text,
            target=self.label_frame_augmentation,
            width=72,
            height=20,
        )

        # 二值化 - 输入框
        self.binaryzation_val = tk.IntVar()
        self.binaryzation_val.set(-1)
        self.binaryzation_entry = ttk.Entry(self.parent, textvariable=self.binaryzation_val, justify=tk.LEFT)
        self.next_to_widget(
            src=self.binaryzation_entry,
            target=self.binaryzation_text,
            width=55,
            height=20,
            tiny_space=True
        )

        # 滤波 - 标签
        self.median_blur_text = ttk.Label(self.parent, text='Median Blur', anchor=tk.W)
        self.next_to_widget(
            src=self.median_blur_text,
            target=self.binaryzation_entry,
            width=80,
            height=20,
            tiny_space=False
        )

        # 滤波 - 输入框
        self.median_blur_val = tk.IntVar()
        self.median_blur_val.set(-1)
        self.median_blur_entry = ttk.Entry(self.parent, textvariable=self.median_blur_val, justify=tk.LEFT)
        self.next_to_widget(
            src=self.median_blur_entry,
            target=self.median_blur_text,
            width=52,
            height=20,
            tiny_space=True
        )

        # 高斯模糊 - 标签
        self.gaussian_blur_text = ttk.Label(self.parent, text='Gaussian Blur', anchor=tk.W)
        self.next_to_widget(
            src=self.gaussian_blur_text,
            target=self.median_blur_entry,
            width=85,
            height=20,
            tiny_space=False
        )

        # 高斯模糊 - 输入框
        self.gaussian_blur_val = tk.IntVar()
        self.gaussian_blur_val.set(-1)
        self.gaussian_blur_entry = ttk.Entry(self.parent, textvariable=self.gaussian_blur_val, justify=tk.LEFT)
        self.next_to_widget(
            src=self.gaussian_blur_entry,
            target=self.gaussian_blur_text,
            width=62,
            height=20,
            tiny_space=True
        )

        # 直方图均衡化 - 多选框
        self.equalize_hist_val = tk.IntVar()
        self.equalize_hist_val.set(0)
        self.equalize_hist = ttk.Checkbutton(
            self.parent, text='EqualizeHist', variable=self.equalize_hist_val, offvalue=0
        )
        self.next_to_widget(
            src=self.equalize_hist,
            target=self.gaussian_blur_entry,
            width=100,
            height=20,
            tiny_space=False
        )

        # 拉普拉斯 - 多选框
        self.laplace_val = tk.IntVar()
        self.laplace_val.set(0)
        self.laplace = ttk.Checkbutton(
            self.parent, text='Laplace', variable=self.laplace_val, onvalue=1, offvalue=0
        )
        self.next_to_widget(
            src=self.laplace,
            target=self.equalize_hist,
            width=64,
            height=20,
            tiny_space=False
        )

        # 旋转 - 标签
        self.rotate_text = ttk.Label(self.parent, text='Rotate (0-90)', anchor=tk.W)
        self.below_widget(
            src=self.rotate_text,
            target=self.binaryzation_text,
            width=72,
            height=20,
            tiny_space=True
        )

        # 旋转 - 输入框
        self.rotate_val = tk.IntVar()
        self.rotate_val.set(-1)
        self.rotate_entry = ttk.Entry(self.parent, textvariable=self.rotate_val, justify=tk.LEFT)
        self.next_to_widget(
            src=self.rotate_entry,
            target=self.rotate_text,
            width=55,
            height=20,
            tiny_space=True
        )

        # 椒盐噪声 - 标签
        self.sp_noise_text = ttk.Label(self.parent, text='Pepper Noise (0-1)', anchor=tk.W)
        self.next_to_widget(
            src=self.sp_noise_text,
            target=self.rotate_entry,
            width=110,
            height=20,
            tiny_space=False
        )

        # 椒盐噪声 - 输入框
        self.sp_noise_val = tk.DoubleVar()
        self.sp_noise_val.set(-1)
        self.sp_noise_entry = ttk.Entry(self.parent, textvariable=self.sp_noise_val, justify=tk.LEFT)
        self.next_to_widget(
            src=self.sp_noise_entry,
            target=self.sp_noise_text,
            width=71,
            height=20,
            tiny_space=True
        )

        # 透视变换 - 多选框
        self.warp_perspective_val = tk.IntVar()
        self.warp_perspective_val.set(0)
        self.warp_perspective = ttk.Checkbutton(
            self.parent, text='Warp Perspective', variable=self.warp_perspective_val, onvalue=1, offvalue=0
        )
        self.next_to_widget(
            src=self.warp_perspective,
            target=self.sp_noise_entry,
            width=130,
            height=20,
            tiny_space=False
        )

        # ============================= Group 5 =====================================
        self.label_frame_project = ttk.Labelframe(self.parent, text='Project Configuration')
        self.below_widget(
            src=self.label_frame_project,
            target=self.label_frame_augmentation,
            width=790,
            height=60,
            tiny_space=True
        )

        # 项目名 - 标签
        self.project_name_text = ttk.Label(self.parent, text='Project Name', anchor=tk.W)
        self.inside_widget(
            src=self.project_name_text,
            target=self.label_frame_project,
            width=90,
            height=20
        )

        # 项目名 - 下拉输入框
        self.comb_project_name = ttk.Combobox(self.parent)
        self.next_to_widget(
            src=self.comb_project_name,
            target=self.project_name_text,
            width=430,
            height=20,
            tiny_space=True
        )
        self.comb_project_name.bind(
            sequence="<Return>",
            func=lambda x: self.project_name_fill_callback(x)
        )
        self.comb_project_name.bind(
            sequence="<Button-1>",
            func=lambda x: self.fetch_projects()
        )
        self.comb_project_name.bind("<<ComboboxSelected>>", lambda x: self.read_conf(x))

        # 保存配置 - 按钮
        self.btn_save_conf = ttk.Button(
            self.parent, text='Save Configuration', command=lambda: self.save_conf()
        )
        self.next_to_widget(
            src=self.btn_save_conf,
            target=self.comb_project_name,
            width=130,
            height=24,
            tiny_space=False,
            offset_y=-2
        )

        # 删除项目 - 按钮
        self.btn_delete = ttk.Button(
            self.parent, text='Delete', command=lambda: self.delete_project()
        )
        self.next_to_widget(
            src=self.btn_delete,
            target=self.btn_save_conf,
            width=80,
            height=24,
            tiny_space=False,
        )

        # ============================= Group 6 =====================================
        self.label_frame_dataset = ttk.Labelframe(
            self.parent, text='Sample Dataset'
        )
        self.below_widget(
            src=self.label_frame_dataset,
            target=self.label_frame_project,
            width=790,
            height=170,
            tiny_space=True
        )

        # 附加训练集 - 按钮
        self.btn_attach_dataset = ttk.Button(
            self.parent,
            text='Attach Dataset',
            command=lambda: self.attach_dataset()
        )
        self.inside_widget(
            src=self.btn_attach_dataset,
            target=self.label_frame_dataset,
            width=120,
            height=24,
        )

        # 附加训练集 - 显示框
        self.attach_dataset_val = tk.StringVar()
        self.attach_dataset_val.set('')
        self.attach_dataset_entry = ttk.Entry(
            self.parent, textvariable=self.attach_dataset_val, justify=tk.LEFT, state=tk.DISABLED
        )
        self.next_to_widget(
            src=self.attach_dataset_entry,
            target=self.btn_attach_dataset,
            width=420,
            height=24,
            tiny_space=True
        )

        # 验证集数目 - 标签
        self.validation_num_text = ttk.Label(self.parent, text='Validation Set Num', anchor=tk.W)
        self.next_to_widget(
            src=self.validation_num_text,
            target=self.attach_dataset_entry,
            width=120,
            height=20,
            tiny_space=False,
            offset_y=2
        )

        # 验证集数目 - 输入框
        self.validation_num_val = tk.IntVar()
        self.validation_num_val.set(300)
        self.validation_num_entry = ttk.Entry(self.parent, textvariable=self.validation_num_val, justify=tk.LEFT)
        self.next_to_widget(
            src=self.validation_num_entry,
            target=self.validation_num_text,
            width=71,
            height=20,
            tiny_space=True
        )

        # 训练集路径 - 标签
        self.dataset_train_path_text = ttk.Label(self.parent, text='Training Dataset', anchor=tk.W)
        self.below_widget(
            src=self.dataset_train_path_text,
            target=self.btn_attach_dataset,
            width=100,
            height=20,
            tiny_space=False
        )

        # 训练集路径 - 列表框
        self.dataset_train_listbox = tk.Listbox(self.parent, font=('微软雅黑', 9))
        self.next_to_widget(
            src=self.dataset_train_listbox,
            target=self.dataset_train_path_text,
            width=640,
            height=36,
            tiny_space=False
        )
        self.dataset_train_listbox.bind(
            sequence="<Delete>",
            func=lambda x: self.listbox_delete_item_callback(x, self.dataset_train_listbox)
        )
        self.listbox_scrollbar(self.dataset_train_listbox)

        # 验证集路径 - 标签
        label_edge = self.object_edge_info(self.dataset_train_path_text)
        widget_edge = self.object_edge_info(self.dataset_train_listbox)
        self.dataset_validation_path_text = ttk.Label(self.parent, text='Validation Dataset', anchor=tk.W)
        self.dataset_validation_path_text.place(
            x=label_edge['x'],
            y=widget_edge['edge_y'] + self.layout['global']['space']['y'] / 2,
            width=100,
            height=20
        )

        # 验证集路径 - 下拉输入框
        self.dataset_validation_listbox = tk.Listbox(self.parent, font=('微软雅黑', 9))
        self.next_to_widget(
            src=self.dataset_validation_listbox,
            target=self.dataset_validation_path_text,
            width=640,
            height=36,
            tiny_space=False
        )
        self.dataset_validation_listbox.bind(
            sequence="<Delete>",
            func=lambda x: self.listbox_delete_item_callback(x, self.dataset_validation_listbox)
        )
        self.listbox_scrollbar(self.dataset_validation_listbox)

        self.sample_map = {
            DatasetType.Directory: {
                RunMode.Trains: self.source_train_path_listbox,
                RunMode.Validation: self.source_validation_path_listbox
            },
            DatasetType.TFRecords: {
                RunMode.Trains: self.dataset_train_listbox,
                RunMode.Validation: self.dataset_validation_listbox
            }
        }

        # 开始训练 - 按钮
        self.btn_training = ttk.Button(self.parent, text='Start Training', command=lambda: self.start_training())
        self.widget_from_right(
            src=self.btn_training,
            target=self.label_frame_dataset,
            width=120,
            height=24,
            tiny_space=True
        )

        # 终止训练 - 按钮
        self.btn_stop = ttk.Button(self.parent, text='Stop', command=lambda: self.stop_training())
        self.button_state(self.btn_stop, tk.DISABLED)
        self.before_widget(
            src=self.btn_stop,
            target=self.btn_training,
            width=60,
            height=24,
            tiny_space=True
        )

        # 编译模型 - 按钮
        self.btn_compile = ttk.Button(self.parent, text='Compile', command=lambda: self.compile())
        self.before_widget(
            src=self.btn_compile,
            target=self.btn_stop,
            width=80,
            height=24,
            tiny_space=True
        )

        # 打包训练集 - 按钮
        self.btn_make_dataset = ttk.Button(self.parent, text='Make Dataset', command=lambda: self.make_dataset())
        self.before_widget(
            src=self.btn_make_dataset,
            target=self.btn_compile,
            width=120,
            height=24,
            tiny_space=True
        )

        # 打包训练集 - 按钮
        self.btn_reset_history = ttk.Button(
            self.parent, text='Reset History', command=lambda: self.reset_history()
        )
        self.before_widget(
            src=self.btn_reset_history,
            target=self.btn_make_dataset,
            width=120,
            height=24,
            tiny_space=True
        )

    def widget_from_right(self, src, target, width, height, tiny_space=False):
        target_edge = self.object_edge_info(target)
        src.place(
            x=self.window_width - width - self.layout['global']['space']['x'],
            y=target_edge['edge_y'] + self.layout['global']['tiny_space' if tiny_space else 'space']['y'],
            width=width,
            height=height
        )

    def before_widget(self, src, target, width, height, tiny_space=False):
        target_edge = self.object_edge_info(target)
        src.place(
            x=target_edge['x'] - width - self.layout['global']['tiny_space' if tiny_space else 'space']['x'],
            y=target_edge['y'],
            width=width,
            height=height
        )

    def inside_widget(self, src, target, width, height):
        target_edge = self.object_edge_info(target)
        src.place(
            x=target_edge['x'] + self.layout['global']['space']['x'],
            y=target_edge['y'] + self.layout['global']['space']['y'],
            width=width,
            height=height
        )

    def below_widget(self, src, target, width, height, tiny_space=False):
        target_edge = self.object_edge_info(target)
        src.place(
            x=target_edge['x'],
            y=target_edge['edge_y'] + self.layout['global']['tiny_space' if tiny_space else 'space']['y'],
            width=width,
            height=height
        )

    def next_to_widget(self, src, target, width, height, tiny_space=False, offset_y=0):
        target_edge = self.object_edge_info(target)
        src.place(
            x=target_edge['edge_x'] + self.layout['global']['tiny_space' if tiny_space else 'space']['x'],
            y=target_edge['y'] + offset_y,
            width=width,
            height=height
        )

    @staticmethod
    def threading_exec(func, *args) -> threading.Thread:
        th = threading.Thread(target=func, args=args)
        th.setDaemon(True)
        th.start()
        return th

    @staticmethod
    def object_edge_info(obj):
        info = obj.place_info()
        x = int(info['x'])
        y = int(info['y'])
        edge_x = int(info['x']) + int(info['width'])
        edge_y = int(info['y']) + int(info['height'])
        return {'x': x, 'y': y, 'edge_x': edge_x, 'edge_y': edge_y}

    @staticmethod
    def listbox_scrollbar(listbox: tk.Listbox):
        y_scrollbar = tk.Scrollbar(
            listbox, command=listbox.yview
        )
        y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        listbox.config(yscrollcommand=y_scrollbar.set)

    def blank_click(self, event):
        if self.current_project != self.comb_project_name.get():
            self.project_name_fill_callback(event)

    def project_name_fill_callback(self, event):
        suffix = '-{}-{}-H{}-{}-C{}'.format(
            self.comb_neu_cnn.get(),
            self.comb_recurrent.get(),
            self.units_num_spin.get(),
            self.comb_loss.get(),
            self.comb_channel.get(),
        )
        current_project_name = self.comb_project_name.get()
        if len(current_project_name) > 0 and current_project_name not in self.project_names:
            self.sample_map[DatasetType.Directory][RunMode.Trains].delete(0, tk.END)
            self.sample_map[DatasetType.Directory][RunMode.Validation].delete(0, tk.END)
            if not current_project_name.endswith(suffix):
                self.comb_project_name.insert(tk.END, suffix)
            self.current_project = self.comb_project_name.get()
            self.update_dataset_files_path(mode=RunMode.Trains)
            self.update_dataset_files_path(mode=RunMode.Validation)

    @property
    def project_path(self):
        if not self.current_project:
            return None
        project_path = "{}/{}".format(self.project_root_path, self.current_project)
        if not os.path.exists(project_path):
            os.makedirs(project_path)
        return project_path

    def update_dataset_files_path(self, mode: RunMode):
        dataset_name = "dataset/{}.0.tfrecords".format(mode.value)
        dataset_path = os.path.join(self.project_path, dataset_name)
        dataset_path = dataset_path.replace("\\", '/')
        self.sample_map[DatasetType.TFRecords][mode].delete(0, tk.END)
        self.sample_map[DatasetType.TFRecords][mode].insert(tk.END, dataset_path)
        self.save_conf()

    def attach_dataset(self):
        if self.is_task_running:
            messagebox.showerror(
                "Error!", "Please terminate the current training first or wait for the training to end."
            )
            return
        if not self.current_project:
            messagebox.showerror(
                "Error!", "Please set the project name first."
            )
            return
        filename = filedialog.askdirectory()
        if not filename:
            return
        model_conf = ModelConfig(self.current_project)

        if not self.check_dataset(model_conf):
            return

        self.attach_dataset_val.set(filename)
        self.button_state(self.btn_attach_dataset, tk.DISABLED)

        for mode in [RunMode.Trains, RunMode.Validation]:
            attached_dataset_name = model_conf.dataset_increasing_name(mode)
            attached_dataset_name = "dataset/{}".format(attached_dataset_name)
            attached_dataset_path = os.path.join(self.project_path, attached_dataset_name)
            attached_dataset_path = attached_dataset_path.replace("\\", '/')
            self.sample_map[DatasetType.TFRecords][mode].insert(tk.END, attached_dataset_path)
        self.save_conf()
        self.threading_exec(
            lambda: DataSets(model_conf).make_dataset(
                trains_path=filename,
                is_add=True,
                callback=lambda: self.button_state(self.btn_attach_dataset, tk.NORMAL),
                msg=lambda x: tk.messagebox.showinfo('Attach Dataset Status', x)
            )
        )
        pass

    @staticmethod
    def button_state(btn: ttk.Button, state: str):
        btn['state'] = state

    def delete_project(self):
        if not self.current_project:
            messagebox.showerror(
                "Error!", "Please select a project to delete."
            )
            return
        if self.is_task_running:
            messagebox.showerror(
                "Error!", "Please terminate the current training first or wait for the training to end."
            )
            return
        project_path = "./projects/{}".format(self.current_project)
        try:
            shutil.rmtree(project_path)
        except Exception as e:
            messagebox.showerror(
                "Error!", json.dumps(e.args)
            )
        messagebox.showinfo(
            "Error!", "Delete successful!"
        )
        self.comb_project_name.delete(0, tk.END)

    def reset_history(self):
        if not self.current_project:
            messagebox.showerror(
                "Error!", "Please select a project first."
            )
            return
        if self.is_task_running:
            messagebox.showerror(
                "Error!", "Please terminate the current training first or wait for the training to end."
            )
            return
        project_history_path = "./projects/{}/model".format(self.current_project)
        try:
            shutil.rmtree(project_history_path)
        except Exception as e:
            messagebox.showerror(
                "Error!", json.dumps(e.args)
            )
        messagebox.showinfo(
            "Error!", "Delete history successful!"
        )

    def auto_loss(self, event):
        if self.comb_recurrent.get() == 'NoRecurrent':
            self.comb_loss.set("CrossEntropy")

    @staticmethod
    def get_param(src: dict, key, default=None):
        result = src.get(key)
        return result if result else default

    def read_conf(self, event):
        selected = self.comb_project_name.get()
        self.current_project = selected
        model_conf = ModelConfig(selected)

        self.size_val.set("[{}, {}]".format(model_conf.image_width, model_conf.image_height))
        self.resize_val.set(json.dumps(model_conf.resize))
        self.source_train_path_listbox.delete(0, tk.END)
        self.source_validation_path_listbox.delete(0, tk.END)
        self.dataset_validation_listbox.delete(0, tk.END)
        self.dataset_train_listbox.delete(0, tk.END)
        for source_train in self.get_param(model_conf.trains_path, DatasetType.Directory, default=[]):
            self.source_train_path_listbox.insert(tk.END, source_train)
        for source_validation in self.get_param(model_conf.validation_path, DatasetType.Directory, default=[]):
            self.source_validation_path_listbox.insert(tk.END, source_validation)
        self.label_num_spin.set(model_conf.max_label_num)
        self.comb_channel.set(model_conf.image_channel)
        self.comb_neu_cnn.set(model_conf.neu_cnn_param)
        self.comb_recurrent.set(model_conf.neu_recurrent_param)
        self.units_num_spin.set(model_conf.units_num)
        self.comb_loss.set(model_conf.loss_func_param)

        if isinstance(model_conf.category_param, list):
            self.category_entry['state'] = tk.NORMAL
            self.comb_category.set('CUSTOMIZED')
            self.category_val.set(json.dumps(model_conf.category_param, ensure_ascii=False))
        else:
            self.category_entry['state'] = tk.DISABLED
            self.comb_category.set(model_conf.category_param)

        self.comb_optimizer.set(model_conf.neu_optimizer_param)
        self.learning_rate_spin.set(model_conf.trains_learning_rate)
        self.end_acc_val.set(model_conf.trains_end_acc)
        self.end_cost_val.set(model_conf.trains_end_cost)
        self.end_epochs_spin.set(model_conf.trains_end_epochs)
        self.batch_size_val.set(model_conf.batch_size)
        self.validation_batch_size_val.set(model_conf.validation_batch_size)
        self.binaryzation_val.set(model_conf.binaryzation)
        self.median_blur_val.set(model_conf.median_blur)
        self.gaussian_blur_val.set(model_conf.gaussian_blur)
        if model_conf.equalize_hist:
            self.equalize_hist_val.set(1)
        if model_conf.laplace:
            self.laplace_val.set(1)
        if model_conf.warp_perspective:
            self.warp_perspective_val.set(1)
        self.rotate_val.set(model_conf.rotate)
        self.sp_noise_val.set(model_conf.sp_noise)
        for dataset_validation in self.get_param(model_conf.validation_path, DatasetType.TFRecords, default=[]):
            self.dataset_validation_listbox.insert(tk.END, dataset_validation)
        for dataset_train in self.get_param(model_conf.trains_path, DatasetType.TFRecords, default=[]):
            self.dataset_train_listbox.insert(tk.END, dataset_train)
        return model_conf

    @property
    def validation_batch_size(self):
        if self.dataset_validation_listbox.size() > 1:
            return self.validation_batch_size_val.get()
        else:
            return min(self.validation_batch_size_val.get(), self.validation_num_val.get())

    def save_conf(self):
        if not self.current_project:
            messagebox.showerror(
                "Error!", "Please set the project name first."
            )
            return
        model_conf = ModelConfig(
            project_name=self.current_project,
            MemoryUsage=0.7,
            CNNNetwork=self.neu_cnn,
            RecurrentNetwork=self.neu_recurrent,
            UnitsNum=self.units_num_spin.get(),
            Optimizer=self.optimizer,
            LossFunction=self.loss_func,
            Decoder=self.comb_loss.get(),
            ModelName=self.current_project,
            ModelField=ModelField.Image.value,
            ModelScene=ModelScene.Classification.value,
            Category=self.category,
            Resize=self.resize,
            ImageChannel=self.comb_channel.get(),
            ImageWidth=self.image_width,
            ImageHeight=self.image_height,
            MaxLabelNum=self.label_num_spin.get(),
            ReplaceTransparent=False,
            HorizontalStitching=False,
            OutputSplit='',
            LabelFrom=LabelFrom.FileName.value,
            ExtractRegex='.*?(?=_)',
            LabelSplit='',
            DatasetTrainsPath=self.dataset_value(
                dataset_type=DatasetType.TFRecords, mode=RunMode.Trains
            ),
            DatasetValidationPath=self.dataset_value(
                dataset_type=DatasetType.TFRecords, mode=RunMode.Validation
            ),
            SourceTrainPath=self.dataset_value(
                dataset_type=DatasetType.Directory, mode=RunMode.Trains
            ),
            SourceValidationPath=self.dataset_value(
                dataset_type=DatasetType.Directory, mode=RunMode.Validation
            ),
            ValidationSetNum=self.validation_num_val.get(),
            SavedSteps=100,
            ValidationSteps=500,
            EndAcc=self.end_acc_val.get(),
            EndCost=self.end_cost_val.get(),
            EndEpochs=self.end_epochs_spin.get(),
            BatchSize=self.batch_size_val.get(),
            ValidationBatchSize=self.validation_batch_size,
            LearningRate=self.learning_rate_spin.get(),
            Binaryzation=self.binaryzation_val.get(),
            MedianBlur=self.median_blur_val.get(),
            GaussianBlur=self.gaussian_blur_val.get(),
            EqualizeHist=self.equalize_hist_val.get(),
            Laplace=self.laplace_val.get(),
            WarpPerspective=self.warp_perspective_val.get(),
            Rotate=self.rotate_val.get(),
            PepperNoise=self.sp_noise_val.get(),
        )
        model_conf.update()
        return model_conf

    def make_dataset(self):
        if not self.current_project:
            messagebox.showerror(
                "Error!", "Please set the project name first."
            )
            return
        if self.is_task_running:
            messagebox.showerror(
                "Error!", "Please terminate the current training first or wait for the training to end."
            )
            return
        self.save_conf()
        self.button_state(self.btn_make_dataset, tk.DISABLED)
        model_conf = ModelConfig(self.current_project)
        train_path = self.dataset_value(DatasetType.Directory, RunMode.Trains)
        validation_path = self.dataset_value(DatasetType.Directory, RunMode.Validation)
        if len(train_path) < 1:
            messagebox.showerror(
                "Error!", "{} Sample set has not been added.".format(RunMode.Trains.value)
            )
            self.button_state(self.btn_make_dataset, tk.NORMAL)
            return
        self.threading_exec(
            lambda: DataSets(model_conf).make_dataset(
                trains_path=train_path,
                validation_path=validation_path,
                is_add=False,
                callback=lambda: self.button_state(self.btn_make_dataset, tk.NORMAL),
                msg=lambda x: tk.messagebox.showinfo('Make Dataset Status', x)
            )
        )

    @property
    def size(self):
        return self.json_filter(self.size_val.get(), int)

    @property
    def image_height(self):
        return self.size[1]

    @property
    def image_width(self):
        return self.size[0]

    @property
    def resize(self):
        return self.json_filter(self.resize_val.get(), int)

    @property
    def neu_cnn(self):
        return self.comb_neu_cnn.get()

    @property
    def neu_recurrent(self):
        return self.comb_recurrent.get()

    @property
    def loss_func(self):
        return self.comb_loss.get()

    @property
    def optimizer(self):
        return self.comb_optimizer.get()

    @staticmethod
    def json_filter(content, item_type):
        if not content:
            messagebox.showerror(
                "Error!", "To select a customized category, you must specify the category set manually."
            )
            return None
        try:
            content = json.loads(content)
        except ValueError as e:
            messagebox.showerror(
                "Error!", "Input must be of type JSON."
            )
            return None
        content = [item_type(i) for i in content]
        return content

    @property
    def category(self):
        comb_selected = self.comb_category.get()
        if not comb_selected:
            messagebox.showerror(
                "Error!", "Please select built-in category or custom category first"
            )
            return None
        if comb_selected == 'CUSTOMIZED':
            category_value = self.category_entry.get()
            category_value = self.json_filter(category_value, str)
        else:
            category_value = comb_selected
        return category_value

    def dataset_value(self, dataset_type: DatasetType, mode: RunMode):
        listbox = self.sample_map[dataset_type][mode]
        value = list(listbox.get(0, listbox.size() - 1))
        return value

    def compile_task(self):
        if not self.current_project:
            messagebox.showerror(
                "Error!", "Please set the project name first."
            )
            return
        model_conf = ModelConfig(project_name=self.current_project)
        if not os.path.exists(model_conf.model_root_path):
            messagebox.showerror(
                "Error", "Model storage folder does not exist."
            )
            return
        if len(os.listdir(model_conf.model_root_path)) < 3:
            messagebox.showerror(
                "Error", "There is no training model record, please train before compiling."
            )
            return
        try:
            self.current_task = Trains(model_conf)
            self.current_task.compile_graph(0)
            status = 'Compile completed'
        except Exception as e:
            messagebox.showerror(
                e.__class__.__name__, json.dumps(e.args)
            )
            status = 'Compile failure'
        tk.messagebox.showinfo('Compile Status', status)

    def compile(self):
        self.job = self.threading_exec(
            lambda: self.compile_task()
        )

    def training_task(self):
        model_conf = ModelConfig(project_name=self.current_project)

        self.current_task = Trains(model_conf)
        try:
            self.button_state(self.btn_training, tk.DISABLED)
            self.button_state(self.btn_stop, tk.NORMAL)
            self.is_task_running = True
            self.current_task.train_process()
            status = 'Training completed'
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror(
                e.__class__.__name__, json.dumps(e.args)
            )
            status = 'Training failure'
        self.button_state(self.btn_training, tk.NORMAL)
        self.button_state(self.btn_stop, tk.DISABLED)
        self.is_task_running = False
        tk.messagebox.showinfo('Training Status', status)

    @staticmethod
    def check_dataset(model_conf):
        trains_path = model_conf.trains_path[DatasetType.TFRecords]
        validation_path = model_conf.validation_path[DatasetType.TFRecords]
        if not trains_path or not validation_path:
            messagebox.showerror(
                "Error!", "Training set or validation set not defined."
            )
            return False
        for tp in trains_path:
            if not os.path.exists(tp):
                messagebox.showerror(
                    "Error!", "Training set path does not exist, please make dataset first"
                )
                return False
        for vp in validation_path:
            if not os.path.exists(vp):
                messagebox.showerror(
                    "Error!", "Validation set path does not exist, please make dataset first"
                )
                return False
        return True

    def start_training(self):
        if not self.check_resize():
            return
        if not self.current_project:
            messagebox.showerror(
                "Error!", "Please set the project name first."
            )
            return
        model_conf = self.save_conf()
        if not self.check_dataset(model_conf):
            return
        self.job = self.threading_exec(
            lambda: self.training_task()
        )

    def stop_training(self):
        self.current_task.stop_flag = True

    @property
    def project_names(self):
        return os.listdir(self.project_root_path)

    def fetch_projects(self):
        self.comb_project_name['values'] = self.project_names

    def browse_dataset(self, dataset_type: DatasetType, mode: RunMode):
        if not self.current_project:
            messagebox.showerror(
                "Error!", "Please define the project name first."
            )
            return
        filename = filedialog.askdirectory()
        if not filename:
            return
        self.sample_map[dataset_type][mode].insert(tk.END, filename)
        self.fetch_sample([filename])

    @staticmethod
    def closest_category(category):
        category = set(category)
        category_group = dict()
        for key in SIMPLE_CATEGORY_MODEL.keys():
            category_set = set(category_extract(key))
            if category <= category_set:
                category_group[key] = len(category_set) - len(category)
        min_index = min(category_group.values())
        for k, v in category_group.items():
            if v == min_index:
                return k

    def fetch_sample(self, dataset_path):
        file_names = os.listdir(dataset_path[0])[0:100]
        category = list()
        len_label = -1

        for file_name in file_names:
            if "_" in file_name:
                label = file_name.split("_")[0]
                label = [i for i in label]
                len_label = len(label)
                category.extend(label)

        category_pram = self.closest_category(category)
        self.comb_category.set(category_pram)
        size = PilImage.open(os.path.join(dataset_path[0], file_names[0])).size
        self.size_val.set(json.dumps(size))
        self.resize_val.set(json.dumps(size))
        self.label_num_spin.set(len_label)

    @staticmethod
    def listbox_delete_item_callback(event, listbox: tk.Listbox):
        i = listbox.curselection()[0]
        listbox.delete(i)

    def comb_category_callback(self, event):
        comb_selected = self.comb_category.get()
        if comb_selected == 'CUSTOMIZED':
            self.category_entry['state'] = tk.NORMAL
        else:
            self.category_entry.delete(0, tk.END)
            self.category_entry['state'] = tk.DISABLED

    def check_resize(self):
        param = OUTPUT_SHAPE1_MAP[NETWORK_MAP[self.neu_cnn]]
        shape1w = math.ceil(1.0*self.resize[0]/param[0])
        shape1h = math.ceil(1.0*self.resize[1]/param[0])
        input_s1 = shape1w * shape1h * param[1]
        label_num = int(self.label_num_spin.get())
        if input_s1 % label_num != 0:
            messagebox.showerror(
                "Error!", "Shape[1] = {} must divide the label_num = {}.".format(input_s1, label_num)
            )
            return False
        return True


    @staticmethod
    def resource_path(relative_path):
        try:
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = sys._MEIPASS
        except AttributeError:
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)


if __name__ == '__main__':
    root = tk.Tk()
    app = Wizard(root)
    root.mainloop()

