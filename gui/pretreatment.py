#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import json
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox
from gui.utils import LayoutGUI


class PretreatmentDialog(tk.Toplevel):

    def __init__(self):
        tk.Toplevel.__init__(self)
        self.title('Data Pretreatment')
        self.layout = {
            'global': {
                'start': {'x': 15, 'y': 20},
                'space': {'x': 15, 'y': 25},
                'tiny_space': {'x': 5, 'y': 10}
            }
        }
        self.pretreatment_entity = None
        self.window_width = 600
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
        self.label_frame_pretreatment = ttk.Labelframe(self, text='Data Pretreatment')
        self.label_frame_pretreatment.place(
            x=self.layout['global']['start']['x'],
            y=self.layout['global']['start']['y'],
            width=575,
            height=150
        )

        # 帧拼接 - 输入框
        self.concat_frames_val = tk.StringVar()
        self.concat_frames_val.set("")
        self.concat_frames_entry = ttk.Entry(self, textvariable=self.concat_frames_val, justify=tk.LEFT)
        self.concat_frames_entry['state'] = tk.DISABLED

        # 帧拼接 - 复选框
        self.concat_frames_check_val = tk.IntVar()
        self.concat_frames_check = ttk.Checkbutton(
            self,
            text='GIF Frame Stitching',
            variable=self.concat_frames_check_val,
            onvalue=1,
            offvalue=0,
            command=lambda: self.check_btn_event(src=self.concat_frames_check_val, entry=self.concat_frames_entry)
        )
        self.layout_utils.inside_widget(
            src=self.concat_frames_check,
            target=self.label_frame_pretreatment,
            width=140,
            height=20,
        )

        # 帧拼接 - 布局
        self.layout_utils.next_to_widget(
            src=self.concat_frames_entry,
            target=self.concat_frames_check,
            width=100,
            height=20,
            tiny_space=True
        )

        # 帧融合 - 输入框
        self.blend_frames_val = tk.StringVar()
        self.blend_frames_val.set("")
        self.blend_frames_entry = ttk.Entry(self, textvariable=self.blend_frames_val, justify=tk.LEFT)
        self.blend_frames_entry['state'] = tk.DISABLED

        # 帧融合 - 复选框
        self.blend_frames_check_val = tk.IntVar()
        self.blend_frames_check_val.set(0)
        self.blend_frames_check = ttk.Checkbutton(
            self, text='GIF Blend Frame',
            variable=self.blend_frames_check_val,
            onvalue=1,
            offvalue=0,
            command=lambda: self.check_btn_event(src=self.blend_frames_check_val, entry=self.blend_frames_entry)
        )

        # 帧融合 - 布局
        self.layout_utils.next_to_widget(
            src=self.blend_frames_check,
            target=self.concat_frames_entry,
            width=120,
            height=20,
            tiny_space=False
        )
        self.layout_utils.next_to_widget(
            src=self.blend_frames_entry,
            target=self.blend_frames_check,
            width=110,
            height=20,
            tiny_space=True
        )

        # 替换透明 - 复选框
        self.replace_transparent_check_val = tk.IntVar()
        self.replace_transparent_check = ttk.Checkbutton(
            self, text='Replace Transparent',
            variable=self.replace_transparent_check_val,
            onvalue=1,
            offvalue=0
        )
        self.layout_utils.below_widget(
            src=self.replace_transparent_check,
            target=self.concat_frames_check,
            width=140,
            height=20,
        )

        # 水平拼接 - 复选框
        self.horizontal_stitching_check_val = tk.IntVar()
        self.horizontal_stitching_check_val.set(0)
        self.horizontal_stitching_check = ttk.Checkbutton(
            self, text='Horizontal Stitching',
            variable=self.horizontal_stitching_check_val,
            onvalue=1,
            offvalue=0
        )
        self.layout_utils.next_to_widget(
            src=self.horizontal_stitching_check,
            target=self.replace_transparent_check,
            width=130,
            height=20,
            tiny_space=False
        )

        # 二值化 - 标签
        self.binaryzation_text = ttk.Label(self, text='Binaryzation', anchor=tk.W)
        self.layout_utils.next_to_widget(
            src=self.binaryzation_text,
            target=self.horizontal_stitching_check,
            width=75,
            height=20,
            tiny_space=False
        )

        # 二值化 - 输入框
        self.binaryzation_val = tk.IntVar()
        self.binaryzation_val.set(-1)
        self.binaryzation_entry = ttk.Entry(self, textvariable=self.binaryzation_val, justify=tk.LEFT)
        self.layout_utils.next_to_widget(
            src=self.binaryzation_entry,
            target=self.binaryzation_text,
            width=55,
            height=20,
            tiny_space=True
        )

        # 保存 - 按钮
        self.btn_save = ttk.Button(self, text='Save Configuration', command=lambda: self.save_conf())
        self.layout_utils.widget_from_right(
            src=self.btn_save,
            target=self.label_frame_pretreatment,
            width=120,
            height=24,
            tiny_space=True
        )

    @staticmethod
    def check_btn_event(src: tk.IntVar, entry: tk.Entry):
        if src.get() == 1:
            entry['state'] = tk.NORMAL
        else:
            entry['state'] = tk.DISABLED
        return None

    def read_conf(self, entity):
        self.pretreatment_entity = entity

        try:

            if entity.blend_frames == -1 or self.blend_frames_entry['state'] == tk.DISABLED:
                self.blend_frames_check_val.set(0)
                self.blend_frames_val.set(json.dumps([-1]))
            else:
                self.blend_frames_check_val.set(1)
                self.blend_frames_entry['state'] = tk.NORMAL
                self.blend_frames_val.set(json.dumps(entity.blend_frames))

            if entity.concat_frames == -1 or self.concat_frames_entry['state'] == tk.DISABLED:
                self.concat_frames_check_val.set(0)
                self.concat_frames_val.set(json.dumps([0, -1]))
            else:
                self.concat_frames_check_val.set(1)
                self.concat_frames_entry['state'] = tk.NORMAL
                self.concat_frames_val.set(json.dumps(entity.concat_frames))

            self.horizontal_stitching_check_val.set(1 if entity.horizontal_stitching else 0)
            self.replace_transparent_check_val.set(1 if entity.replace_transparent else 0)

            self.binaryzation_val.set(entity.binaryzation)

        except Exception as e:
            messagebox.showerror(
                e.__class__.__name__, json.dumps(e.args)
            )
            return

    def save_conf(self):
        try:

            if self.concat_frames_check_val.get() == 1:
                self.pretreatment_entity.concat_frames = json.loads(self.concat_frames_val.get())
            else:
                self.pretreatment_entity.concat_frames = -1
            if self.blend_frames_check_val.get() == 1:
                self.pretreatment_entity.blend_frames = json.loads(self.blend_frames_val.get())
            else:
                self.pretreatment_entity.blend_frames = -1
            self.pretreatment_entity.horizontal_stitching = True if self.horizontal_stitching_check_val.get() == 1 else False
            self.pretreatment_entity.replace_transparent = True if self.replace_transparent_check_val.get() == 1 else False
            self.pretreatment_entity.binaryzation = self.binaryzation_val.get()
        except Exception as e:
            messagebox.showerror(
                e.__class__.__name__, json.dumps(e.args)
            )
            return

        self.destroy()