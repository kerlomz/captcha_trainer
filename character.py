#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>

SPACE_INDEX = 0
SPACE_TOKEN = ['']
NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ALPHA_UPPER = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
               'V', 'W', 'X', 'Y', 'Z']
ALPHA_LOWER = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
               'v', 'w', 'x', 'y', 'z']
OPERATOR = ['(', ')', 'P', 'S', 'M', 'D']
FLOAT = ['.']

SIMPLE_CHAR_SET = dict(
    NUMERIC=NUMBER,
    ALPHANUMERIC=NUMBER + ALPHA_LOWER + ALPHA_UPPER,
    ALPHANUMERIC_LOWER=NUMBER + ALPHA_LOWER,
    ALPHANUMERIC_UPPER=NUMBER + ALPHA_UPPER,
    ALPHABET_LOWER=ALPHA_LOWER,
    ALPHABET_UPPER=ALPHA_UPPER,
    ALPHABET=ALPHA_LOWER + ALPHA_UPPER,
    OPERATION=NUMBER + OPERATOR,
    FLOAT=NUMBER + FLOAT
)
