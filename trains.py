#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import time
import logging
import numpy as np
import tensorflow as tf
import framework_lstm
import utils
from config import *
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger('Training for OCR using CNN+LSTM+CTC')
logger.setLevel(logging.INFO)


def compile_graph(sess, acc):
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    for node in input_graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr:
                del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr:
                del node.attr['use_locking']

    output_graph_def = convert_variables_to_constants(
        sess,
        input_graph_def,
        output_node_names=['lstm/output/predict']
    )

    last_compile_model_path = COMPILE_MODEL_PATH.replace('.pb', '_{}.pb'.format(int(acc * 10000)))
    with tf.gfile.FastGFile(last_compile_model_path, mode='wb') as gf:
        gf.write(output_graph_def.SerializeToString())


def train_process(mode=RunMode.Trains):
    model = framework_lstm.LSTM(mode)
    model.build_graph()

    print('Loading Trains DataSet...')
    train_feeder = utils.DataIterator(mode=RunMode.Trains)
    if TRAINS_USE_TFRECORDS:
        train_feeder.read_sample_from_tfrecords()
    else:
        train_feeder.read_sample_from_files()
    print('Total {} Trains DataSets'.format(train_feeder.size))

    print('Loading Test DataSet...')
    test_feeder = utils.DataIterator(mode=RunMode.Test)
    if TEST_USE_TFRECORDS:
        test_feeder.read_sample_from_tfrecords()
    else:
        test_feeder.read_sample_from_files()
    print('Total {} Test DataSets'.format(test_feeder.size))

    num_train_samples = train_feeder.size
    num_batches_per_epoch = int(num_train_samples / BATCH_SIZE)

    num_val_samples = test_feeder.size
    num_batches_per_epoch_val = int(num_val_samples / BATCH_SIZE)
    shuffle_idx_val = np.random.permutation(num_val_samples)

    config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        gpu_options=tf.GPUOptions(
            allow_growth=True,  # it will cause fragmentation.
            per_process_gpu_memory_fraction=GPU_USAGE)
    )
    accuracy = 0
    with tf.Session(config=config) as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
        train_writer = tf.summary.FileWriter('logs', sess.graph)
        try:
            saver.restore(sess, tf.train.latest_checkpoint(MODEL_PATH))
        except ValueError:
            pass

        print('Start training...')

        while 1:
            shuffle_idx = np.random.permutation(num_train_samples)
            train_cost = 0
            epoch_count = 1
            start_time = time.time()

            for cur_batch in range(num_batches_per_epoch):
                index_list = [
                    shuffle_idx[i % num_train_samples] for i in
                    range(cur_batch * BATCH_SIZE, (cur_batch + 1) * BATCH_SIZE)
                ]
                if TRAINS_USE_TFRECORDS:
                    batch_inputs, _, batch_labels = train_feeder.generate_batch_by_tfrecords(sess)
                else:
                    batch_inputs, _, batch_labels = train_feeder.generate_batch_by_index(index_list)

                feed = {
                    model.batch_size: BATCH_SIZE,
                    model.inputs: batch_inputs,
                    model.labels: batch_labels,
                }

                summary_str, batch_cost, step, _ = sess.run(
                    [model.merged_summary, model.cost, model.global_step, model.train_op],
                    feed
                )
                train_cost += batch_cost * BATCH_SIZE
                train_writer.add_summary(summary_str, step)

                if step % TRAINS_SAVE_STEPS == 0:
                    saver.save(sess, SAVE_MODEL, global_step=step)
                    logger.info('save checkpoint at step {0}', format(step))

                if step % TRAINS_VALIDATION_STEPS == 0:
                    acc_batch_total = 0
                    lr = 0
                    batch_time = time.time()

                    for j in range(num_batches_per_epoch_val):
                        index_val = [
                            shuffle_idx_val[i % num_val_samples] for i in
                            range(j * BATCH_SIZE, (j + 1) * BATCH_SIZE)
                        ]
                        if TRAINS_USE_TFRECORDS:
                            val_inputs, _, val_labels = test_feeder.generate_batch_by_tfrecords(sess)
                        else:
                            val_inputs, _, val_labels = test_feeder.generate_batch_by_index(index_val)

                        val_feed = {
                            model.batch_size: BATCH_SIZE,
                            model.inputs: val_inputs,
                            model.labels: val_labels,
                        }

                        dense_decoded, last_batch_err, lr = sess.run(
                            [model.dense_decoded, model.cost, model.lrn_rate],
                            val_feed
                        )
                        if TRAINS_USE_TFRECORDS:
                            ori_labels = test_feeder.label_by_tfrecords()
                        else:
                            ori_labels = test_feeder.label_by_index(index_val)
                        acc = utils.accuracy_calculation(
                            ori_labels,
                            dense_decoded,
                            ignore_value=-1,
                        )
                        acc_batch_total += acc

                    accuracy = (acc_batch_total * BATCH_SIZE) / num_val_samples
                    avg_train_cost = train_cost / ((cur_batch + 1) * BATCH_SIZE)

                    log = "Epoch: {}, Step: {} Accuracy = {:.3f}, Cost = {:.3f}, Time = {:.3f}, LearningRate: {}"
                    print(log.format(epoch_count, step, accuracy, avg_train_cost, time.time() - batch_time, lr))
                    if accuracy >= TRAINS_END_ACC and epoch_count >= TRAINS_END_EPOCHS:
                        break
            if accuracy >= TRAINS_END_ACC:
                compile_graph(sess, accuracy)
                print('Total Time: {}'.format(time.time() - start_time))
                break
            epoch_count += 1

        coord.request_stop()
        coord.join(threads)


def main(_):
    init()
    train_process()
    print('Training completed.')
    pass


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
