#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import sys
import time


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
    # raise SystemException(text, code)
    Error(text, code)


class ConfigException:
    SYS_CONFIG_PATH_NOT_EXIST = -4041
    MODEL_CONFIG_PATH_NOT_EXIST = -4042
    CHAR_SET_NOT_EXIST = -4043
    CHAR_SET_INCORRECT = -4043
    SAMPLE_LABEL_ERROR = -4044
    GET_LABEL_REGEX_ERROR = -4045
    INSUFFICIENT_SAMPLE = -5
    TESTSET_SIZE_ERROR = -6

