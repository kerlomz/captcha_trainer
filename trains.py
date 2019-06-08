#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import time
import random
import numpy as np
import tensorflow as tf
import framework
import utils
from config import *
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
tf.logging.set_verbosity(tf.logging.INFO)


def compile_graph(acc):
    input_graph = tf.Graph()
    sess = tf.Session(graph=input_graph)

    with sess.graph.as_default():
        model = framework.GraphOCR(
            RunMode.Predict,
            NETWORK_MAP[NEU_CNN],
            NETWORK_MAP[NEU_RECURRENT]
        )
        model.build_graph()
        input_graph_def = sess.graph.as_graph_def()
        saver = tf.train.Saver(var_list=tf.global_variables())
        tf.logging.info(tf.train.latest_checkpoint(MODEL_PATH))
        saver.restore(sess, tf.train.latest_checkpoint(MODEL_PATH))

    output_graph_def = convert_variables_to_constants(
        sess,
        input_graph_def,
        output_node_names=['dense_decoded']
    )

    last_compile_model_path = COMPILE_MODEL_PATH.replace('.pb', '_{}.pb'.format(int(acc * 10000)))
    with tf.gfile.GFile(last_compile_model_path, mode='wb') as gf:
        gf.write(output_graph_def.SerializeToString())

    generate_config(acc)


def train_process(mode=RunMode.Trains):
    model = framework.GraphOCR(mode, NETWORK_MAP[NEU_CNN], NETWORK_MAP[NEU_RECURRENT])
    model.build_graph()

    tf.logging.info('Loading Trains DataSet...')
    train_feeder = utils.DataIterator(mode=RunMode.Trains)
    if TRAINS_USE_TFRECORDS:
        train_feeder.read_sample_from_tfrecords(TRAINS_PATH)
        tf.logging.info('Loading Test DataSet...')
        test_feeder = utils.DataIterator(mode=RunMode.Test)
        test_feeder.read_sample_from_tfrecords(TEST_PATH)
    else:
        if isinstance(TRAINS_PATH, list):
            origin_list = []
            for trains_path in TRAINS_PATH:
                origin_list += [os.path.join(trains_path, trains) for trains in os.listdir(trains_path)]
        else:
            origin_list = [os.path.join(TRAINS_PATH, trains) for trains in os.listdir(TRAINS_PATH)]
        np.random.shuffle(origin_list)
        if not HAS_TEST_SET:
            test_list = origin_list[:TEST_SET_NUM]
            trains_list = origin_list[TEST_SET_NUM:]
        else:
            if isinstance(TEST_PATH, list):
                test_list = []
                for test_path in TEST_PATH:
                    test_list += [os.path.join(test_path, test) for test in os.listdir(test_path)]
            else:
                test_list = [os.path.join(TEST_PATH, test) for test in os.listdir(TEST_PATH)]
            np.random.shuffle(test_list)
            trains_list = origin_list
        train_feeder.read_sample_from_files(trains_list)
        tf.logging.info('Loading Test DataSet...')
        test_feeder = utils.DataIterator(mode=RunMode.Test)
        test_feeder.read_sample_from_files(test_list)

    tf.logging.info('Total {} Trains DataSets'.format(train_feeder.size))
    tf.logging.info('Total {} Test DataSets'.format(test_feeder.size))
    if test_feeder.size >= train_feeder.size:
        exception("The number of training sets cannot be less than the test set.", )

    num_train_samples = train_feeder.size
    num_test_samples = test_feeder.size
    if num_test_samples < TEST_BATCH_SIZE:
        exception(
            "The number of test sets cannot be less than the test batch size.",
            ConfigException.INSUFFICIENT_SAMPLE
        )
    num_batches_per_epoch = int(num_train_samples / BATCH_SIZE)

    config = tf.ConfigProto(
        # allow_soft_placement=True,
        log_device_placement=False,
        gpu_options=tf.GPUOptions(
            allocator_type='BFC',
            allow_growth=True,  # it will cause fragmentation.
            per_process_gpu_memory_fraction=GPU_USAGE)
    )
    accuracy = 0
    epoch_count = 1

    with tf.Session(config=config) as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
        train_writer = tf.summary.FileWriter('logs', sess.graph)
        try:
            saver.restore(sess, tf.train.latest_checkpoint(MODEL_PATH))
        except ValueError:
            pass
        tf.logging.info('Start training...')

        while 1:
            shuffle_trains_idx = np.random.permutation(num_train_samples)
            start_time = time.time()
            last_train_avg_cost = 0
            for cur_batch in range(num_batches_per_epoch):
                batch_time = time.time()
                index_list = [
                    shuffle_trains_idx[i % num_train_samples] for i in
                    range(cur_batch * BATCH_SIZE, (cur_batch + 1) * BATCH_SIZE)
                ]
                if TRAINS_USE_TFRECORDS:
                    classified_batch = train_feeder.generate_batch_by_tfrecords(sess)
                else:
                    classified_batch = train_feeder.generate_batch_by_files(index_list)
                step = 0
                class_num = len(classified_batch)
                avg_cost = 0
                for index, (shape, batch) in enumerate(classified_batch.items()):
                    batch_inputs, batch_seq_len, batch_labels = batch
                    feed = {
                        model.inputs: batch_inputs,
                        model.labels: batch_labels,
                    }

                    summary_str, batch_cost, step, _ = sess.run(
                        [model.merged_summary, model.cost, model.global_step, model.train_op],
                        feed_dict=feed
                    )
                    avg_cost += batch_cost
                    last_train_avg_cost = avg_cost / class_num
                    train_writer.add_summary(summary_str, step)
                    if step % 100 == index and step not in range(class_num):
                        tf.logging.info('Step: {} Time: {:.3f} sec/batch, Cost = {:.5f}, {}-BatchSize: {}'.format(
                            step,
                            time.time() - batch_time,
                            batch_cost,
                            shape,
                            len(batch_inputs)
                        ))
                    if step % TRAINS_SAVE_STEPS == index and index == (class_num - 1) and step not in range(class_num):
                        saver.save(sess, SAVE_MODEL, global_step=step)
                        # tf.logging.info('save checkpoint at step {0}'.format(step))

                if step % TRAINS_VALIDATION_STEPS == (class_num - 1) and step not in range(class_num):
                    shuffle_test_idx = np.random.permutation(num_test_samples)
                    batch_time = time.time()
                    index_test = [
                        shuffle_test_idx[i % num_test_samples] for i in
                        range(cur_batch * TEST_BATCH_SIZE, (cur_batch + 1) * TEST_BATCH_SIZE)
                    ]
                    if TRAINS_USE_TFRECORDS:
                        classified_batch = test_feeder.generate_batch_by_tfrecords(sess)
                    else:
                        classified_batch = test_feeder.generate_batch_by_files(index_test)

                    all_dense_decoded = []
                    lr = 0

                    for index, (shape, batch) in enumerate(classified_batch.items()):
                        test_inputs, batch_seq_len, test_labels = batch
                        val_feed = {
                            model.inputs: test_inputs,
                            model.labels: test_labels
                        }
                        dense_decoded, sub_lr = sess.run(
                            [model.dense_decoded, model.lrn_rate],
                            feed_dict=val_feed
                        )
                        all_dense_decoded += dense_decoded.tolist()
                        lr += sub_lr
                    accuracy = utils.accuracy_calculation(
                        test_feeder.labels,
                        all_dense_decoded,
                        ignore_value=[0, -1],
                    )
                    log = "Epoch: {}, Step: {}, Accuracy = {:.4f}, Cost = {:.5f}, " \
                          "Time = {:.3f} sec/batch, LearningRate: {}"
                    tf.logging.info(log.format(
                        epoch_count,
                        step,
                        accuracy,
                        last_train_avg_cost, time.time() - batch_time, lr / len(classified_batch)
                    ))

                    if accuracy >= TRAINS_END_ACC and epoch_count >= TRAINS_END_EPOCHS and last_train_avg_cost <= TRAINS_END_COST:
                        break
            if accuracy >= TRAINS_END_ACC and epoch_count >= TRAINS_END_EPOCHS and last_train_avg_cost <= TRAINS_END_COST:
                compile_graph(accuracy)
                tf.logging.info('Total Time: {} sec.'.format(time.time() - start_time))
                break
            epoch_count += 1


def generate_config(acc):
    with open(MODEL_CONFIG_PATH, "r", encoding="utf8") as current_fp:
        text = "".join(current_fp.readlines())
        text = text.replace("ModelName: {}".format(TARGET_MODEL), "ModelName: {}_{}".format(TARGET_MODEL, int(acc * 10000)))
    with open(os.path.join(OUTPUT_PATH, "{}_model.yaml".format(TARGET_MODEL)), "w", encoding="utf8") as save_fp:
        save_fp.write(text)


def main(_):
    init()
    train_process()
    tf.logging.info('Training completed.')
    pass


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
