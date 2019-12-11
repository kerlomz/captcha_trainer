#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import sys
import time

"""
此类包含各种异常类别，希望对已知可能的异常进行分类，以便出现问题是方便定位
"""


class SystemException(RuntimeError):
    def __init__(self, message, code=-1):
        self.message = message
        self.code = code


class Error(object):
    def __init__(self, message, code=-1):
        self.message = message
        self.code = code
        print(self.message)
        time.sleep(5)
        sys.exit(self.code)


def exception(text, code=-1):
    raise SystemException(text, code)
    # Error(text, code)


class ConfigException:
    OPTIMIZER_NOT_SUPPORTED = -4072
    NETWORK_NOT_SUPPORTED = -4071
    LOSS_FUNC_NOT_SUPPORTED = -4061
    MODEL_FIELD_NOT_SUPPORTED = -4052
    MODEL_SCENE_NOT_SUPPORTED = -4051
    SYS_CONFIG_PATH_NOT_EXIST = -4041
    MODEL_CONFIG_PATH_NOT_EXIST = -4042
    CATEGORY_NOT_EXIST = -4043
    CATEGORY_INCORRECT = -4043
    SAMPLE_LABEL_ERROR = -4044
    GET_LABEL_REGEX_ERROR = -4045
    ERROR_LABEL_FROM = -4046
    INSUFFICIENT_SAMPLE = -5
    VALIDATION_SET_SIZE_ERROR = -6


