#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import time
from PyInstaller.__main__ import run
from config import resource_path


with open("../resource/VERSION", "w", encoding="utf8") as f:
    today = time.strftime("%Y%m%d", time.localtime(time.time()))
    f.write(today)


def package(prefix):
    """基于PyInstaller打包编译为单可执行文件"""
    opts = ['{}app.spec'.format(prefix), '--distpath={}dist'.format(prefix), '--workpath={}build'.format(prefix)]
    run(opts)


if __name__ == '__main__':
    try:
        package("../")
    except FileNotFoundError:
        package("/")
