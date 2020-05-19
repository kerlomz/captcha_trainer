#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>

import tensorflow as tf
import core
import utils
import utils.data
import validation
from config import *
from tf_graph_util import convert_variables_to_constants
from PIL import ImageFile
from tf_onnx_util import convert_onnx

ImageFile.LOAD_TRUNCATED_IMAGES = True
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)


class Trains:

    stop_flag: bool = False
    """训练任务的类"""

    def __init__(self, model_conf: ModelConfig):
        """
        :param model_conf: 读取工程配置文件
        """
        self.model_conf = model_conf
        self.validation = validation.Validation(self.model_conf)

    @staticmethod
    def compile_onnx(predict_sess, output_graph_def, input_path):
        convert_onnx(
            sess=predict_sess,
            graph_def=output_graph_def,
            input_path=input_path,
            inputs_op="input:0",
            outputs_op="dense_decoded:0"
        )

    @staticmethod
    def compile_tflite(input_path):
        input_tensor_name = ["input"]
        classes_tensor_name = ["dense_decoded"]

        converter = tf.lite.TFLiteConverter.from_frozen_graph(
            input_path,
            input_tensor_name,
            classes_tensor_name,
        )
        # converter.post_training_quantize = True
        tflite_model = converter.convert()
        output_path = input_path.replace(".pb", ".tflite")
        with open(output_path, "wb") as f:
            f.write(tflite_model)

    def compile_graph(self, acc):
        """
        编译当前准确率下对应的计算图为pb模型，准确率仅作为模型命名的一部分
        :param acc: 准确率
        :return:
        """
        input_graph = tf.Graph()
        predict_sess = tf.Session(graph=input_graph)

        with predict_sess.graph.as_default():
            model = core.NeuralNetwork(
                model_conf=self.model_conf,
                mode=RunMode.Predict,
                backbone=self.model_conf.neu_cnn,
                recurrent=self.model_conf.neu_recurrent
            )
            model.build_graph()
            input_graph_def = predict_sess.graph.as_graph_def()
            saver = tf.train.Saver(var_list=tf.global_variables())
            tf.logging.info(tf.train.latest_checkpoint(self.model_conf.model_root_path))
            saver.restore(predict_sess, tf.train.latest_checkpoint(self.model_conf.model_root_path))

            output_graph_def = convert_variables_to_constants(
                predict_sess,
                input_graph_def,
                output_node_names=['dense_decoded']
            )

        if not os.path.exists(self.model_conf.compile_model_path):
            os.makedirs(self.model_conf.compile_model_path)

        last_compile_model_path = (
            os.path.join(self.model_conf.compile_model_path, "{}.pb".format(self.model_conf.model_name))
        ).replace('.pb', '_{}.pb'.format(int(acc * 10000)))

        self.model_conf.output_config(target_model_name="{}_{}".format(self.model_conf.model_name, int(acc * 10000)))
        with tf.io.gfile.GFile(last_compile_model_path, mode='wb') as gf:
            gf.write(output_graph_def.SerializeToString())

        if self.model_conf.loss_func == LossFunction.CrossEntropy:
            self.compile_onnx(predict_sess, output_graph_def, last_compile_model_path)
        # self.compile_tflite(last_compile_model_path)

    def achieve_cond(self, acc, cost, epoch):
        achieve_accuracy = acc >= self.model_conf.trains_end_acc
        achieve_cost = cost <= self.model_conf.trains_end_cost
        achieve_epochs = epoch >= self.model_conf.trains_end_epochs
        over_epochs = epoch > 10000
        if (achieve_accuracy and achieve_epochs and achieve_cost) or over_epochs:
            return True
        return False

    def train_process(self):
        """
        训练任务
        :return:
        """
        # 输出重要的配置参数
        self.model_conf.println()
        # 定义网络结构
        model = core.NeuralNetwork(
            mode=RunMode.Trains,
            model_conf=self.model_conf,
            backbone=self.model_conf.neu_cnn,
            recurrent=self.model_conf.neu_recurrent
        )
        model.build_graph()

        tf.compat.v1.logging.info('Loading Trains DataSet...')
        train_feeder = utils.data.DataIterator(model_conf=self.model_conf, mode=RunMode.Trains)
        train_feeder.read_sample_from_tfrecords(self.model_conf.trains_path[DatasetType.TFRecords])

        tf.compat.v1.logging.info('Loading Validation DataSet...')
        validation_feeder = utils.data.DataIterator(model_conf=self.model_conf, mode=RunMode.Validation)
        validation_feeder.read_sample_from_tfrecords(self.model_conf.validation_path[DatasetType.TFRecords])

        tf.logging.info('Total {} Trains DataSets'.format(train_feeder.size))
        tf.logging.info('Total {} Validation DataSets'.format(validation_feeder.size))
        if validation_feeder.size >= train_feeder.size:
            exception("The number of training sets cannot be less than the test set.", )

        num_train_samples = train_feeder.size
        num_validation_samples = validation_feeder.size

        if num_validation_samples < self.model_conf.validation_batch_size:
            self.model_conf.validation_batch_size = num_validation_samples
            tf.logging.warn(
                'The number of validation sets is less than the validation batch size, '
                'will use validation set size as validation batch size.'.format(validation_feeder.size))

        num_batches_per_epoch = int(num_train_samples / self.model_conf.batch_size)
        # 会话配置
        sess_config = tf.compat.v1.ConfigProto(
            # allow_soft_placement=True,
            log_device_placement=False,
            gpu_options=tf.compat.v1.GPUOptions(
                allocator_type='BFC',
                allow_growth=True,  # it will cause fragmentation.
                per_process_gpu_memory_fraction=self.model_conf.memory_usage)
        )
        accuracy = 0
        epoch_count = 1

        if num_train_samples < 500:
            save_step = 10
            trains_validation_steps = 50

        else:
            save_step = 100
            trains_validation_steps = self.model_conf.trains_validation_steps

        with tf.compat.v1.Session(config=sess_config) as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=2)
            train_writer = tf.compat.v1.summary.FileWriter('logs', sess.graph)
            # try:
            checkpoint_state = tf.train.get_checkpoint_state(self.model_conf.model_root_path)
            if checkpoint_state and checkpoint_state.model_checkpoint_path:
                # 加载被中断的训练任务
                saver.restore(sess, checkpoint_state.model_checkpoint_path)

            tf.logging.info('Start training...')

            # 进入训练任务循环
            while 1:

                start_time = time.time()
                batch_cost = 65535
                # 批次循环
                for cur_batch in range(num_batches_per_epoch):

                    if self.stop_flag:
                        break

                    batch_time = time.time()

                    trains_batch = train_feeder.generate_batch_by_tfrecords(sess)

                    batch_inputs, batch_labels = trains_batch

                    feed = {
                        model.inputs: batch_inputs,
                        model.labels: batch_labels,
                        model.utils.is_training: True
                    }

                    summary_str, batch_cost, step, _, seq_len = sess.run(
                        [model.merged_summary, model.cost, model.global_step, model.train_op, model.seq_len],
                        feed_dict=feed
                    )
                    train_writer.add_summary(summary_str, step)

                    if step % save_step == 0 and step != 0:
                        tf.logging.info(
                            'Step: {} Time: {:.3f} sec/batch, Cost = {:.8f}, BatchSize: {}, Shape[1]: {}'.format(
                                step,
                                time.time() - batch_time,
                                batch_cost,
                                len(batch_inputs),
                                seq_len[0]
                            )
                        )

                    # 达到保存步数对模型过程进行存储
                    if step % trains_validation_steps == 0 and step != 0:
                        saver.save(sess, self.model_conf.save_model, global_step=step)

                    # 进入验证集验证环节
                    if step % trains_validation_steps == 0 and step != 0:

                        batch_time = time.time()
                        validation_batch = validation_feeder.generate_batch_by_tfrecords(sess)

                        test_inputs, test_labels = validation_batch
                        val_feed = {
                            model.inputs: test_inputs,
                            model.labels: test_labels,
                            model.utils.is_training: False
                        }
                        dense_decoded, lr = sess.run(
                            [model.dense_decoded, model.lrn_rate],
                            feed_dict=val_feed
                        )
                        # 计算准确率
                        accuracy = self.validation.accuracy_calculation(
                            validation_feeder.labels,
                            dense_decoded,
                        )
                        log = "Epoch: {}, Step: {}, Accuracy = {:.4f}, Cost = {:.5f}, " \
                              "Time = {:.3f} sec/batch, LearningRate: {}"
                        tf.logging.info(log.format(
                            epoch_count,
                            step,
                            accuracy,
                            batch_cost,
                            time.time() - batch_time,
                            lr / len(validation_batch),
                        ))

                        # 满足终止条件但尚未完成当前epoch时跳出epoch循环
                        if self.achieve_cond(acc=accuracy, cost=batch_cost, epoch=epoch_count):
                            break

                # 满足终止条件时，跳出任务循环
                if self.stop_flag:
                    break
                if self.achieve_cond(acc=accuracy, cost=batch_cost, epoch=epoch_count):
                    self.compile_graph(accuracy)
                    tf.logging.info('Total Time: {} sec.'.format(time.time() - start_time))
                    break
                epoch_count += 1


def main(argv):
    project_name = argv[-1]
    model_conf = ModelConfig(project_name=project_name)
    Trains(model_conf).train_process()
    tf.logging.info('Training completed.')
    pass


if __name__ == '__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    tf.compat.v1.app.run()
