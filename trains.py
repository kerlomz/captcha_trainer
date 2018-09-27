#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import time
from utils import *
from PIL import Image
from predict import predict_func
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from framework_cnn import *
from tools.quantize import quantize

TRAINS_GROUP = path2list(TRAINS_PATH, True)
TEST_GROUP = path2list(TEST_PATH, True)

LAST_COMPILE_MODEL_PATH = ""


def compile_graph(sess, input_graph_def, acc):
    output_graph_def = convert_variables_to_constants(
        sess,
        input_graph_def,
        output_node_names=['output/predict']
    )
    global LAST_COMPILE_MODEL_PATH
    LAST_COMPILE_MODEL_PATH = COMPILE_MODEL_PATH.replace('.pb', '_{}.pb'.format(int(acc * 10000)))
    with tf.gfile.FastGFile(LAST_COMPILE_MODEL_PATH, mode='wb') as gf:
        gf.write(output_graph_def.SerializeToString())


def train_process():
    _network = CNN().network()
    global_step = tf.Variable(0, trainable=False)

    _label = tf.reshape(label, [-1, MAX_CAPTCHA_LEN, CHAR_SET_LEN])
    max_idx_predict = _network['predict']
    max_idx_label = tf.argmax(_label, 2)
    correct_predict = tf.equal(max_idx_predict, max_idx_label)

    with tf.name_scope('monitor'):
        loss = tf.reduce_mean(
            # Loss Function
            tf.nn.sigmoid_cross_entropy_with_logits(logits=_network['final_output'], labels=_label)
        )
    tf.summary.scalar('loss', loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=TRAINS_LEARNING_RATE).minimize(loss, global_step=global_step)

    with tf.name_scope('monitor'):
        accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    with tf.device(DEVICE):
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            gpu_options=tf.GPUOptions(
                allow_growth=True,  # it will cause fragmentation.
                per_process_gpu_memory_fraction=GPU_USAGE))
        )
    print('Session Initializing...')
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('logs', sess.graph)
    # Save the training process
    print('Loading history archive...')
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
    try:
        saver.restore(sess, tf.train.latest_checkpoint(MODEL_PATH))
    except ValueError:
        pass
    print('Initialized.\n---------------------------------------------------------------------------------')
    time_epoch_start = time.time()
    index = 0

    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    merged = tf.summary.merge_all()
    while True:

        batch_x, batch_y = get_next_batch(64, False)  # 64
        summary, _, _loss, step = sess.run(
            [merged, optimizer, loss, global_step],
            feed_dict={
                x: batch_x,
                label: batch_y,
                keep_prob: 0.95
            }
        )
        writer.add_summary(summary, index)
        index += 1

        if step % 10 == 0:
            print('Loss: {}'.format(step), _loss)

        if step % TRAINS_SAVE_STEP == 0:
            saver.save(sess, SAVE_MODEL, global_step=step)
        else:
            continue

        epoch_time = time.time() - time_epoch_start
        time_epoch_start = time.time()
        acc = test_training(sess, max_idx_predict)
        print(
            'Epoch Spend: %0.2fs' % epoch_time,
            'Test Predict Accuracy Rate: %0.2f%%' % (acc * 100)
        )
        if acc > COMPILE_ACC:
            compile_graph(sess, input_graph_def, acc)

        if acc > TRAINS_END_ACC and step > TRAINS_END_STEP:
            compile_graph(sess, input_graph_def, acc)
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
    origin_size = pil_image.size
    define_size = RESIZE if RESIZE else (IMAGE_WIDTH, IMAGE_HEIGHT)
    if define_size != origin_size:
        pil_image = pil_image.resize(define_size)
    define_size = (origin_size[0] * MAGNIFICATION, origin_size[1] * MAGNIFICATION)
    if define_size != origin_size:
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
            # Output specific correct label
            # print("Flag: {}  Predict: {}".format(text, predict_text))
            right_cnt += 1
        else:
            pass
            # Output specific error labels
            # print(
            #     "False, Label: {}  Predict: {}".format(text, predict_text)
            # )
    return right_cnt / task_cnt


if __name__ == '__main__':
    init()
    train_process()
    print('Training completed.')
    quantize(LAST_COMPILE_MODEL_PATH, QUANTIZED_MODEL_PATH)
    pass
