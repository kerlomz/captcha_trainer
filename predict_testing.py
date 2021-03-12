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

# argv = sys.argv[1]


class Predict:
    def __init__(self, project_name):
        self.model_conf = ModelConfig(project_name=project_name)
        self.encoder = Encoder(model_conf=self.model_conf, mode=RunMode.Predict)

    def get_image_batch(self, img_bytes):
        if not img_bytes:
            return []
        return [self.encoder.image(index) for index in [img_bytes]]

    @staticmethod
    def decode_maps(categories):
        """解码器"""
        return {index: category for index, category in enumerate(categories, 0)}

    def predict_func(self, image_batch, _sess, dense_decoded, op_input):
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
                if class_index == -1 or class_index == self.model_conf.category_num:
                    expression += ''
                else:
                    expression += self.decode_maps(self.model_conf.category)[class_index]
            decoded_expression.append(expression)
        return ''.join(decoded_expression) if len(decoded_expression) > 1 else decoded_expression[0]

    def testing(self, image_dir, limit=None):

        graph = tf.Graph()
        sess = tf.compat.v1.Session(
            graph=graph,
            config=tf.compat.v1.ConfigProto(
                # allow_soft_placement=True,
                # log_device_placement=True,
                gpu_options=tf.compat.v1.GPUOptions(
                    allocator_type='BFC',
                    # allow_growth=True,  # it will cause fragmentation.
                    per_process_gpu_memory_fraction=0.1
                ))
        )

        with sess.graph.as_default():

            sess.run(tf.compat.v1.global_variables_initializer())
            # tf.keras.backend.set_session(session=sess)

            model = NeuralNetwork(
                self.model_conf,
                RunMode.Predict,
                self.model_conf.neu_cnn,
                self.model_conf.neu_recurrent
            )
            model.build_graph()
            model.build_train_op()

            saver = tf.compat.v1.train.Saver(var_list=tf.compat.v1.global_variables())

            """从项目中加载最后一次训练的网络参数"""
            saver.restore(sess, tf.train.latest_checkpoint(self.model_conf.model_root_path))
            # model.build_graph()
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
        dir_list = os.listdir(image_dir)
        random.shuffle(dir_list)
        lines = []
        for i, p in enumerate(dir_list):
            n = os.path.join(image_dir, p)
            if limit and i > limit:
                break
            with open(n, "rb") as f:
                b = f.read()

            batch = self.get_image_batch(b)
            if not batch:
                continue
            st = time.time()
            predict_text = self.predict_func(
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
            # is_mark = '_' in p
            # p = p.replace("\\", "/")
            label = re.search(self.model_conf.extract_regex, p.split(PATH_SPLIT)[-1])
            label = label.group() if label else p.split(".")[0]
            # if is_mark:
            if 'LOWER' in self.model_conf.category_param:
                label = label.lower()
                t = label == predict_text.lower()
            elif 'UPPER' in self.model_conf.category_param:
                label = label.upper()
                t = label == predict_text.upper()
            else:
                t = label == predict_text
            # Used to verify test sets
            if t:
                true_count += 1
            else:
                false_count += 1
            print(i, p, label, predict_text, t, true_count / (true_count + false_count), (et-st) * 1000)
            # else:
            #     print(i, p, predict_text, true_count / (true_count + false_count), (et - st) * 1000)
            # with open("competition_format.csv", "w", encoding="utf8") as f:
            #     f.write("\n".join(lines))
        sess.close()


if __name__ == '__main__':

    predict = Predict(project_name=sys.argv[1])
    predict.testing(image_dir=r"H:\TrainSet\*", limit=None)


