#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import time
import random
import logging
import numpy as np
import tensorflow as tf
import framework
import utils
from config import *
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger('Training for OCR using {}+{}+CTC'.format(NEU_CNN, NEU_RECURRENT))
logger.setLevel(logging.INFO)


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
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(MODEL_PATH))

    output_graph_def = convert_variables_to_constants(
        sess,
        input_graph_def,
        output_node_names=['dense_decoded']
    )

    last_compile_model_path = COMPILE_MODEL_PATH.replace('.pb', '_{}.pb'.format(int(acc * 10000)))
    with tf.gfile.FastGFile(last_compile_model_path, mode='wb') as gf:
        gf.write(output_graph_def.SerializeToString())


def train_process(mode=RunMode.Trains):
    model = framework.GraphOCR(mode, NETWORK_MAP[NEU_CNN], NETWORK_MAP[NEU_RECURRENT])
    model.build_graph()
    test_list, trains_list = None, None
    if not HAS_TEST_SET:
        trains_list = os.listdir(TRAINS_PATH)
        random.shuffle(trains_list)
        origin_list = [os.path.join(TRAINS_PATH, trains) for i, trains in enumerate(trains_list)]
        test_list = origin_list[:TEST_SET_NUM]
        trains_list = origin_list[TEST_SET_NUM:]

    print('Loading Trains DataSet...')
    train_feeder = utils.DataIterator(mode=RunMode.Trains)
    if TRAINS_USE_TFRECORDS:
        train_feeder.read_sample_from_tfrecords()
    else:
        train_feeder.read_sample_from_files(trains_list)
    print('Total {} Trains DataSets'.format(train_feeder.size))

    print('Loading Test DataSet...')
    test_feeder = utils.DataIterator(mode=RunMode.Test)
    if TEST_USE_TFRECORDS:
        test_feeder.read_sample_from_tfrecords()
    else:
        test_feeder.read_sample_from_files(test_list)
    print('Total {} Test DataSets'.format(test_feeder.size))

    num_train_samples = train_feeder.size
    num_batches_per_epoch = int(num_train_samples / BATCH_SIZE)

    config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        gpu_options=tf.GPUOptions(
            # allow_growth=True,  # it will cause fragmentation.
            per_process_gpu_memory_fraction=GPU_USAGE)
    )
    accuracy = 0
    epoch_count = 1

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
            start_time = time.time()

            if TRAINS_USE_TFRECORDS:
                test_inputs, batch_seq_len, test_labels = test_feeder.generate_batch_by_tfrecords(sess)
            else:
                test_inputs, batch_seq_len, test_labels = test_feeder.generate_batch_by_files()

            val_feed = {
                model.inputs: test_inputs,
                model.labels: test_labels
            }

            for cur_batch in range(num_batches_per_epoch):
                batch_time = time.time()
                index_list = [
                    shuffle_idx[i % num_train_samples] for i in
                    range(cur_batch * BATCH_SIZE, (cur_batch + 1) * BATCH_SIZE)
                ]
                if TRAINS_USE_TFRECORDS:
                    batch_inputs, batch_seq_len, batch_labels = train_feeder.generate_batch_by_tfrecords(sess)
                else:
                    batch_inputs, batch_seq_len, batch_labels = train_feeder.generate_batch_by_files(index_list)

                feed = {
                    model.inputs: batch_inputs,
                    model.labels: batch_labels,
                }

                summary_str, batch_cost, step, _ = sess.run(
                    [model.merged_summary, model.cost, model.global_step, model.train_op],
                    feed
                )
                train_cost += batch_cost * BATCH_SIZE
                avg_train_cost = train_cost / ((cur_batch + 1) * BATCH_SIZE)

                train_writer.add_summary(summary_str, step)

                if step % 100 == 0:
                    print('Step: {} Time: {:.3f}, Cost = {:.3f}'.format(step, time.time() - batch_time, avg_train_cost))

                if step % TRAINS_SAVE_STEPS == 0:
                    saver.save(sess, SAVE_MODEL, global_step=step)
                    logger.info('save checkpoint at step {0}', format(step))

                if step % TRAINS_VALIDATION_STEPS == 0:
                    batch_time = time.time()

                    dense_decoded, last_batch_err, lr = sess.run(
                        [model.dense_decoded, model.last_batch_error, model.lrn_rate],
                        val_feed
                    )

                    accuracy = utils.accuracy_calculation(
                        test_feeder.labels(),
                        dense_decoded,
                        ignore_value=-1,
                    )
                    log = "Epoch: {}, Step: {}, Accuracy = {:.3f}, Cost = {:.3f}, " \
                          "Time = {:.3f}, LearningRate: {}, LastBatchError: {}"
                    print(log.format(
                        epoch_count, step, accuracy, avg_train_cost, time.time() - batch_time, lr, last_batch_err
                    ))

                    if accuracy >= TRAINS_END_ACC and epoch_count >= TRAINS_END_EPOCHS:
                        break
            if accuracy >= TRAINS_END_ACC and epoch_count >= TRAINS_END_EPOCHS:
                compile_graph(accuracy)
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
