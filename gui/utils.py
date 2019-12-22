#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>


class LayoutGUI(object):

    def __init__(self, layout, window_width):
        self.layout = layout
        self.window_width = window_width

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

    @staticmethod
    def object_edge_info(obj):
        info = obj.place_info()
        x = int(info['x'])
        y = int(info['y'])
        edge_x = int(info['x']) + int(info['width'])
        edge_y = int(info['y']) + int(info['height'])
        return {'x': x, 'y': y, 'edge_x': edge_x, 'edge_y': edge_y}

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

