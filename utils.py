#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import cv2
import numpy as np
import PIL.Image
import tensorflow as tf
from config import *
from pretreatment import preprocessing

PATH_MAP = {
    RunMode.Trains: TRAINS_PATH,
    RunMode.Test: TEST_PATH
}

REGEX_MAP = {
    RunMode.Trains: TRAINS_REGEX,
    RunMode.Test: TEST_REGEX
}


def encode_maps():
    return {char: i for i, char in enumerate(GEN_CHAR_SET, 0)}


class DataIterator:
    def __init__(self, mode: RunMode):
        self.mode = mode
        self.data_dir = PATH_MAP[mode]
        self.image = []
        self.image_path = []
        self.label_list = []
        self.image_batch = []
        self.label_batch = []
        self._label_batch = []
        self._size = 0

    @staticmethod
    def _encoder(code):
        if isinstance(code, bytes):
            code = code.decode('utf8')

        for k, v in CHAR_REPLACE.items():
            if not k or not v:
                break
            code.replace(k, v)

        code = code.lower() if 'LOWER' in CHAR_SET else code
        code = code.upper() if 'UPPER' in CHAR_SET else code
        return [encode_maps()[c] for c in list(code)]

    def read_sample_from_files(self, data_set=None):
        if data_set:
            self.image_path = data_set
            self.label_list = [
                self._encoder(re.search(REGEX_MAP[self.mode], i.split(PATH_SPLIT)[-1]).group()) for i in data_set
            ]
        else:
            for root, sub_folder, file_list in os.walk(self.data_dir):
                for file_path in file_list:
                    image_name = os.path.join(root, file_path)
                    self.image_path.append(image_name)
                    # Get the label from the file name based on the regular expression.
                    code = re.search(
                        REGEX_MAP[self.mode], image_name.split(PATH_SPLIT)[-1]
                    ).group()
                    # The manual verification code platform is not case sensitive,
                    # - it will affect the accuracy of the training set.
                    # Here is a case conversion based on the selected character set.
                    self.label_list.append(self._encoder(code))
        self._size = len(self.label_list)

    def read_sample_from_tfrecords(self):
        filename_queue = tf.train.string_input_producer([
            os.path.join(TFRECORDS_DIR, TFRECORDS_NAME_MAP[self.mode]+".tfrecords")
        ])
        reader = tf.TFRecordReader()

        self._size = len(
            [_ for _ in tf.python_io.tf_record_iterator(
                os.path.join(TFRECORDS_DIR, TFRECORDS_NAME_MAP[self.mode] + ".tfrecords")
            )]
        )

        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'label': tf.FixedLenFeature([], tf.string),
                'image': tf.FixedLenFeature([], tf.string),
            }
        )
        image = tf.decode_raw(features['image'], tf.uint8)
        image = tf.reshape(image, [IMAGE_HEIGHT, IMAGE_WIDTH, 1])
        label = tf.cast(features['label'], tf.string)

        min_after_dequeue = 1000
        capacity = min_after_dequeue + 3 * BATCH_SIZE
        image_batch, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=BATCH_SIZE,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue
        )
        self.image_batch = image_batch
        self.label_batch = label_batch

    @property
    def size(self):
        return self._size

    def label_by_index(self, index_list):
        return [self.label_list[i] for i in index_list]

    def label_by_tfrecords(self):
        return self._label_batch

    @staticmethod
    def _image(path):

        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # The OpenCV cannot handle gif format images, it will return None.
        if im is None:
            pil_image = PIL.Image.open(path).convert("RGB")
            im = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2GRAY)
        im = preprocessing(im, BINARYZATION, SMOOTH, BLUR).astype(np.float32) / 255.
        im = cv2.resize(im, (IMAGE_WIDTH, IMAGE_HEIGHT))
        return np.reshape(im, [IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    @staticmethod
    def _get_input_lens(sequences):
        lengths = np.asarray([len(_) for _ in sequences], dtype=np.int64)
        return sequences, lengths

    def generate_batch_by_index(self, index):
        image_batch = [self._image(self.image_path[i]) for i in index]
        label_batch = [self.label_list[i] for i in index]
        return self._generate_batch(image_batch, label_batch)

    def _generate_batch(self, image_batch, label_batch):
        batch_inputs, batch_seq_len = self._get_input_lens(np.array(image_batch))
        batch_labels = sparse_tuple_from_label(label_batch)
        return batch_inputs, batch_seq_len, batch_labels

    def generate_batch_by_tfrecords(self, sess):
        _image, _label = sess.run([self.image_batch, self.label_batch])
        image_batch = [i.astype(np.float32) / 255. for i in _image]
        label_batch = [self._encoder(i) for i in _label]
        self._label_batch = label_batch
        return self._generate_batch(image_batch, label_batch)


def accuracy_calculation(original_seq, decoded_seq, ignore_value=-1):
    if len(original_seq) != len(decoded_seq):
        print('original lengths is different from the decoded_seq, please check again')
        return 0
    count = 0
    for i, origin_label in enumerate(original_seq):
        decoded_label = [j for j in decoded_seq[i] if j != ignore_value]
        if origin_label == decoded_label:
            count += 1

    return count * 1.0 / len(original_seq)


# Convert a sequence list to a sparse matrix
def sparse_tuple_from_label(sequences, dtype=np.int32):
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
    return indices, values, shape
