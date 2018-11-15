#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
# import string
from config import *

alphanumeric = char_set("ALPHANUMERIC")
alphanumeric_lower = char_set("ALPHANUMERIC_LOWER")


def check_charset(detail=False, case_sensitive=False, use_config=True):
    recommended_set = ""
    case_insensitive = not case_sensitive
    defined_lower = "LOWER" in CHAR_SET and use_config
    defined_upper = "UPPER" in CHAR_SET and use_config
    lower_case = case_insensitive or defined_lower
    upper_case = case_sensitive or defined_upper
    samples = []
    for trains_path in TRAINS_PATH:
        samples += [trains for trains in os.listdir(trains_path)]
    # samples = os.listdir(TRAINS_PATH)
    letters = char_set(CHAR_SET) if use_config else (alphanumeric if case_sensitive else alphanumeric_lower)
    trains_set_labels = [re.search(TRAINS_REGEX, i).group() for i in samples]
    char_exclude = []
    labels = "".join(trains_set_labels)
    labels = labels if case_sensitive else labels.lower()
    upper_case = True if [i for i in labels if i in ALPHA_UPPER] and upper_case else False
    lower_case = True if [i for i in labels if i in ALPHA_LOWER] and lower_case else False
    number_case = True if [i for i in labels if i in NUMBER] else False

    if not upper_case and lower_case and number_case:
        recommended_set = "ALPHANUMERIC_LOWER"
    elif not upper_case and not number_case and lower_case:
        recommended_set = "ALPHABET_LOWER"
    elif upper_case and number_case and not lower_case:
        recommended_set = "ALPHANUMERIC_UPPER"
    elif upper_case and not number_case and not lower_case:
        recommended_set = "ALPHABET_UPPER"
    elif not lower_case and not upper_case and number_case:
        recommended_set = "NUMERIC"

    for letter in letters:
        labels = labels if case_sensitive else labels.lower()
        count = labels.count(letter)
        if count == 0:
            char_exclude.append(letter)
        if detail:
            print("{}: {}".format(letter, count))

    recommended_set_list = char_set(recommended_set)
    smart_exclude = [i for i in char_exclude if i in recommended_set_list]
    print("Your current character set is {}.".format(CHAR_SET))
    if char_exclude:
        print("We found out from the training set that these characters did not appear:", ", ".join(char_exclude))

    def output(exclude):
        exclude_text = "and keep the default value of the \"CharExclude\" parameter []"
        exclude_text = "and use {} as the \"CharExclude\" parameter".format(smart_exclude, exclude) if exclude else exclude_text
        print(exclude_text)
        print("We recommend that you use \"{}\" as the \"CharSet\" parameter ".format(recommended_set), exclude_text)

    output(smart_exclude)


if __name__ == '__main__':
    check_charset(detail=True, case_sensitive=False, use_config=False)
