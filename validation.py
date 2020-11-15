#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import json
import numpy as np
import tensorflow as tf
from config import ModelConfig


class Validation(object):
    """验证类，用于准确率计算"""
    def __init__(self, model: ModelConfig):
        """
        :param model: 读取配置文件获取当前工程的重要参数：category_num, category
        """
        self.model = model
        self.category_num = self.model.category_num
        self.category = self.model.category

    def accuracy_calculation(self, original_seq, decoded_seq):
        """
        准确率计算函数
        :param original_seq: 密集数组-Y标签
        :param decoded_seq: 密集数组-预测标签
        :return:
        """
        if isinstance(decoded_seq, np.ndarray):
            decoded_seq = decoded_seq.tolist()

        ignore_value = [-1, self.category_num, 0]
        original_seq_len = len(original_seq)
        decoded_seq_len = len(decoded_seq)

        if original_seq_len != decoded_seq_len:
            tf.compat.v1.logging.error(original_seq)
            tf.compat.v1.logging.error(decoded_seq)
            tf.compat.v1.logging.error('original lengths {} is different from the decoded_seq {}, please check again'.format(
                original_seq_len,
                decoded_seq_len
            ))
            return 0
        count = 0

        # Here is for debugging, positioning error source use
        error_sample = []
        for i, origin_label in enumerate(original_seq):

            decoded_label = decoded_seq[i]
            if isinstance(decoded_label, int):
                decoded_label = [decoded_label]
            processed_decoded_label = [j for j in decoded_label if j not in ignore_value]
            processed_origin_label = [j for j in origin_label if j not in ignore_value]

            if i < 5:
                tf.compat.v1.logging.info(
                    "{} {} {} {} {} --> {} {}".format(
                        i,
                        len(processed_origin_label),
                        len(processed_decoded_label),
                        origin_label,
                        decoded_label,
                        [self.category[_] if _ != self.category_num else '-' for _ in origin_label if _ != -1],
                        [self.category[_] if _ != self.category_num else '-' for _ in decoded_label if _ != -1]
                    )
                )
            if processed_origin_label == processed_decoded_label:
                count += 1
            # Training is not useful for decoding
            # Here is for debugging, positioning error source use
            if processed_origin_label != processed_decoded_label and len(error_sample) < 5:
                error_sample.append({
                    "origin": "".join([self.category[_] if _ != self.category_num else '-' for _ in origin_label if _ != -1]),
                    "decode": "".join([self.category[_] if _ != self.category_num else '-' for _ in decoded_label if _ != -1])
                })
        tf.compat.v1.logging.error(json.dumps(error_sample, ensure_ascii=False))
        return count * 1.0 / len(original_seq)