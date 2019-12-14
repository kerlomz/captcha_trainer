#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
"""此脚本用于训练过程中检验训练效果的脚本，功能为：通过启动参数加载【工程名】中的网络进行预测"""
import random
import numpy as np
import tensorflow as tf
from config import *
from constants import RunMode
from encoder import Encoder
from core import NeuralNetwork

project_name = sys.argv[1]

model_conf = ModelConfig(project_name=project_name)
encoder = Encoder(model_conf=model_conf, mode=RunMode.Predict)


def get_image_batch(img_bytes):
    return [encoder.image(index) for index in [img_bytes]]


def decode_maps(categories):
    """解码器"""
    return {index: category for index, category in enumerate(categories, 0)}


def predict_func(image_batch, _sess, dense_decoded, op_input):
    """预测函数"""
    dense_decoded_code = _sess.run(dense_decoded, feed_dict={
        op_input: image_batch,
    })
    # print(dense_decoded_code)
    decoded_expression = []
    for item in dense_decoded_code:
        expression = ''
        # print(item)
        if isinstance(item, int) or isinstance(item, np.int64):
            item = [item]
        for class_index in item:
            if class_index == -1 or class_index == model_conf.category_num:
                expression += ''
            else:
                expression += decode_maps(model_conf.category)[class_index]
        decoded_expression.append(expression)
    return ''.join(decoded_expression) if len(decoded_expression) > 1 else decoded_expression[0]


if __name__ == '__main__':

    """构建计算图"""
    graph = tf.Graph()
    tf_checkpoint = tf.train.latest_checkpoint(model_conf.model_root_path)
    sess = tf.Session(
        graph=graph,
        config=tf.ConfigProto(
            # allow_soft_placement=True,
            # log_device_placement=True,
            gpu_options=tf.GPUOptions(
                allocator_type='BFC',
                # allow_growth=True,  # it will cause fragmentation.
                per_process_gpu_memory_fraction=0.1
            ))
    )
    graph_def = graph.as_graph_def()

    with sess.graph.as_default():

        sess.run(tf.global_variables_initializer())
        tf.keras.backend.set_session(session=sess)
        # with tf.gfile.GFile(COMPILE_MODEL_PATH.replace('.pb', '_{}.pb'.format(int(0.95 * 10000))), "rb") as f:
        #     graph_def_file = f.read()
        # graph_def.ParseFromString(graph_def_file)
        # print('{}.meta'.format(tf_checkpoint))

        model = NeuralNetwork(
            model_conf,
            RunMode.Predict,
            model_conf.neu_cnn,
            model_conf.neu_recurrent
        )
        model.build_graph()

        saver = tf.train.Saver(var_list=tf.global_variables())

        """从项目中加载最后一次训练的网络参数"""
        saver.restore(sess, tf.train.latest_checkpoint(model_conf.model_root_path))

        # _ = tf.import_graph_def(graph_def, name="")

    """定义操作符"""
    dense_decoded_op = sess.graph.get_tensor_by_name("dense_decoded:0")
    x_op = sess.graph.get_tensor_by_name('input:0')
    """固定网络"""
    sess.graph.finalize()

    true_count = 0
    false_count = 0
    """
    以下为根据路径调用预测函数输出结果的demo
    """
    # Fill in your own sample path
    image_dir = r"H:\Task\*"
    dir_list = os.listdir(image_dir)
    random.shuffle(dir_list)
    lines = []
    for i, p in enumerate(dir_list):
        n = os.path.join(image_dir, p)
        # if i > 10000:
        #     break
        with open(n, "rb") as f:
            b = f.read()

        batch = get_image_batch(b)
        st = time.time()
        predict_text = predict_func(
            batch,
            sess,
            dense_decoded_op,
            x_op,
        )
        et = time.time()
        # t = p.split(".")[0].lower() == predict_text.lower()
        # csv_output = "{},{}".format(p.split(".")[0], predict_text)
        # lines.append(csv_output)
        # print(csv_output)

        # Used to verify test sets
        t = p.split("_")[0].lower() == predict_text.lower()
        if t:
            true_count += 1
        else:
            false_count += 1
        print(i, p, p.split("_")[0].lower(), predict_text, t, true_count / (true_count + false_count), (et-st) * 1000)
    # with open("competition_format.csv", "w", encoding="utf8") as f:
    #     f.write("\n".join(lines))



