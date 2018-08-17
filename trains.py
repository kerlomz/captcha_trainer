#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import time
from config import *
from utils import *
from PIL import Image
from predict import predict_func
from tensorflow.python.framework.graph_util import convert_variables_to_constants

TRAINS_GROUP = path2list(TRAINS_PATH, True)
TEST_GROUP = path2list(TEST_PATH, True)

if NEU_NAME == 'DenseNet':
    from framework_densenet import *
else:
    from framework_cnn import *


def train_process():
    _network = DenseNet().network() if NEU_NAME == 'DenseNet' else CNN().network()
    global_step = tf.Variable(0, trainable=False)

    _label = tf.reshape(label, [-1, MAX_CAPTCHA_LEN, CHAR_SET_LEN])
    max_idx_p = _network['predict']  # shape:batch_size, 4, nb_cls
    max_idx_l = tf.argmax(_label, 1 if NEU_NAME == 'DenseNet' else 2)
    correct_predict = tf.equal(max_idx_p, max_idx_l)

    with tf.name_scope('monitor'):
        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=_network['final_output'], labels=_label)
            if NEU_NAME == 'DenseNet' else
            tf.nn.sigmoid_cross_entropy_with_logits(logits=_network['final_output'], labels=_label)
        )
    tf.summary.scalar('loss', loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=TRAINS_LEARNING_RATE).minimize(loss, global_step=global_step)

    with tf.name_scope('monitor'):
        accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    with tf.device(DEVICE):
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('logs', sess.graph)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)  # 将训练过程进行保存
    try:
        saver.restore(sess, tf.train.latest_checkpoint(MODEL_PATH))
    except Exception as e:
        print(e)
    time_epoch_start = time.time()
    index = 0

    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    merged = tf.summary.merge_all()
    while True:

        batch_x, batch_y = get_next_batch(128, False)  # 64
        summary, _, _loss, step = sess.run(
            [merged, optimizer, loss, global_step],
            feed_dict={
                x: batch_x,
                label: batch_y,
                training_flag: True
            } if NEU_NAME == 'DenseNet' else {
                x: batch_x,
                label: batch_y,
                keep_prob: 0.95
            }
        )
        writer.add_summary(summary, index)
        index += 1

        if step % 10 == 0:
            print(step, '损失: ' if LANGUAGE == 'zh-CN' else 'Loss: ', _loss)

        if step % TRAINS_SAVE_STEP == 0:
            saver.save(sess, SAVE_MODEL, global_step=step)
        else:
            continue

        # batch_x_test, batch_y_test = get_next_batch(TRAINS_TEST_NUM, True)
        # acc = sess.run(accuracy, feed_dict={x: batch_x_test, label: batch_y_test, keep_prob: 1.})
        # print(step, '精度评估:\t' if LANGUAGE == 'zh-CN' else 'Train-Acc:\t', acc)
        epoch_time = time.time() - time_epoch_start
        time_epoch_start = time.time()
        acc = test_training(sess, max_idx_p)
        print(
            '迭代时间: %0.2fs' % epoch_time if LANGUAGE == 'zh-CN' else 'EP Spend: %0.2fs' % epoch_time,
            '精度评估: %0.2f%%' % (acc * 100) if LANGUAGE == 'zh-CN' else 'Train-Acc: %0.2f%%' % (
                    acc * 100)
        )
        if acc > 0.9:
            output_graph_def = convert_variables_to_constants(
                sess,
                input_graph_def,
                output_node_names=['output/predict']
            )
            with tf.gfile.FastGFile(COMPILE_MODEL_PATH.replace('.pb', '_{}.pb'.format(int(acc*100))), mode='wb') as gf:
                gf.write(output_graph_def.SerializeToString())

        if acc > TRAINS_END_ACC and step > TRAINS_END_STEP:
            output_graph_def = convert_variables_to_constants(
                sess,
                input_graph_def,
                output_node_names=['output/predict']
            )
            with tf.gfile.FastGFile(COMPILE_MODEL_PATH, mode='wb') as gf:
                gf.write(output_graph_def.SerializeToString())
            break


def get_next_batch(batch_size=64, test=False):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA_LEN * CHAR_SET_LEN])

    for i in range(batch_size):
        text, image = text_and_image(test)
        if 'UPPER' in CHAR_SET:
            text = text.upper()
        elif 'LOWER' in CHAR_SET:
            text = text.lower()
        batch_x[i, :] = image.flatten() / 255
        batch_y[i, :] = text2vec(text)

    return batch_x, batch_y


def text_and_image(test=False):
    global TEST_GROUP, TRAINS_GROUP
    if test:
        TEST_GROUP = path2list(TEST_PATH, True)
    file_list = TEST_GROUP if test else TRAINS_GROUP

    index = random.randint(0, len(file_list) - 1)
    f_path = file_list[index]
    captcha_text = re.search(TEST_REGEX if test else TRAINS_REGEX, f_path.split(PATH_SPLIT)[-1]).group()
    pil_image = Image.open(f_path)
    define_size = RESIZE if RESIZE else (IMAGE_WIDTH, IMAGE_HEIGHT)

    if define_size != pil_image.size:
        pil_image = pil_image.resize(define_size)

    captcha_image = preprocessing(
        pil_image,
        binaryzation=BINARYZATION,
        smooth=SMOOTH,
        blur=BLUR,
        original_color=IMAGE_ORIGINAL_COLOR,
        invert=INVERT
    )
    return captcha_text, captcha_image


def test_training(sess, predict):
    right_cnt = 0
    task_cnt = TRAINS_TEST_NUM
    for i in range(task_cnt):
        text, image = text_and_image(True)
        if 'UPPER' in CHAR_SET:
            text = text.upper()
        elif 'LOWER' in CHAR_SET:
            text = text.lower()
        image = image.flatten() / 255
        predict_text = predict_func(image, sess, predict, x, keep_prob)
        if text == predict_text:
            # print("Flag: {}  Predict: {}".format(text, predict_text))
            right_cnt += 1
        else:
            pass
            # print(
            #     "预测错误, 标注: {}  预测: {}".format(text, predict_text)
            #     if LANGUAGE == 'zh-CN' else
            #     "False, Label: {}  Predict: {}".format(text, predict_text)
            # )
    return right_cnt / task_cnt


if __name__ == '__main__':
    train_process()
    print('end')
    pass
