#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import io
import PIL.Image
import cv2
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from config import *
from constants import RunMode
from pretreatment import preprocessing

PATH_MAP = {
    RunMode.Trains: TRAINS_PATH,
    RunMode.Test: TEST_PATH
}


def encode_maps():
    return {char: i for i, char in enumerate(GEN_CHAR_SET, 0)}


# Training is not useful for decoding
# Here is for debugging, positioning error source use
# def decode_maps():
#     return {i: char for i, char in enumerate(GEN_CHAR_SET, 0)}


class DataIterator:
    def __init__(self, mode: RunMode):
        self.mode = mode
        self.data_dir = PATH_MAP[mode]
        self.next_element = None
        self.image_path = []
        self.label_list = []
        self._label_list = []
        self._size = 0
        self.max_length = 0
        self.is_first = True

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
        try:
            return [encode_maps()[c] for c in list(code)]
        except KeyError as e:
            exception(
                'The sample label {} contains invalid charset: {}.'.format(
                    code, e.args[0]
                ), ConfigException.SAMPLE_LABEL_ERROR
            )

    def read_sample_from_files(self, data_set=None):
        if data_set:
            self.image_path = data_set
            try:
                self._label_list = [
                    self._encoder(re.search(TRAINS_REGEX, i.split(PATH_SPLIT)[-1]).group()) for i in data_set
                ]
            except AttributeError as e:
                regex_not_found = "group" in e.args[0]
                if regex_not_found:
                    exception(
                        "Configured {} is '{}', it may be wrong and unable to get label properly.".format(
                            "TrainRegex",
                            TRAINS_REGEX
                        ),
                        ConfigException.GET_LABEL_REGEX_ERROR
                    )
        else:
            for root, sub_folder, file_list in os.walk(self.data_dir):
                for file_path in file_list:
                    image_name = os.path.join(root, file_path)
                    if file_path in IGNORE_FILES:
                        continue
                    self.image_path.append(image_name)
                    # Get the label from the file name based on the regular expression.
                    code = re.search(
                        TRAINS_REGEX, image_name.split(PATH_SPLIT)[-1]
                    )
                    if not code:
                        exception(
                            "Configured {} is '{}', it may be wrong and unable to get label properly.".format(
                                "TrainRegex",
                                TRAINS_REGEX
                            ),
                            ConfigException.GET_LABEL_REGEX_ERROR
                        )
                    code = code.group()
                    # The manual verification code platform is not case sensitive,
                    # - it will affect the accuracy of the training set.
                    # Here is a case conversion based on the selected character set.
                    self._label_list.append(self._encoder(code))
        self._size = len(self._label_list)

    @staticmethod
    def parse_example(serial_example):

        features = tf.parse_single_example(
            serial_example,
            features={
                'label': tf.FixedLenFeature([], tf.string),
                'image': tf.FixedLenFeature([], tf.string),
            }
        )
        image = tf.cast(features['image'], tf.string)
        label = tf.cast(features['label'], tf.string)

        return image, label

    def read_sample_from_tfrecords(self, path):
        self._size = len([_ for _ in tf.python_io.tf_record_iterator(path)])

        min_after_dequeue = 1000
        batch = BATCH_SIZE if self.mode == RunMode.Trains else TEST_BATCH_SIZE

        dataset_train = tf.data.TFRecordDataset(
            filenames=path,
            # num_parallel_reads=20
        ).map(self.parse_example)
        dataset_train = dataset_train.shuffle(
            min_after_dequeue
        ).batch(batch).repeat()
        iterator = dataset_train.make_one_shot_iterator()
        self.next_element = iterator.get_next()

    @property
    def size(self):
        return self._size

    @property
    def labels(self):
        return self.label_list

    @staticmethod
    def _image(path_or_bytes, is_random=False):

        # im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # The OpenCV cannot handle gif format images, it will return None.
        # if im is None:
        path_or_stream = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes
        pil_image = PIL.Image.open(path_or_stream)
        rgb = pil_image.split()
        size = pil_image.size

        if len(rgb) > 3 and REPLACE_TRANSPARENT:
            background = PIL.Image.new('RGB', pil_image.size, (255, 255, 255))
            background.paste(pil_image, (0, 0, size[0], size[1]), pil_image)
            pil_image = background

        if IMAGE_CHANNEL == 1:
            pil_image = pil_image.convert('L')

        im = np.array(pil_image)
        im = preprocessing(im, BINARYZATION, SMOOTH, BLUR).astype(np.float32)

        if RESIZE[0] == -1:
            random_ratio = random.choice([2.5, 3, 3.5, 3.2, 2.7, 2.75])
            ratio = RESIZE[1] / size[1]
            random_width = int(random_ratio * RESIZE[1])
            resize_width = int(ratio * size[0])
            resize_width = random_width if is_random else resize_width
            im = cv2.resize(im, (resize_width, RESIZE[1]))
        else:
            im = cv2.resize(im, (RESIZE[0], RESIZE[1]))
        im = im.swapaxes(0, 1)
        return np.array((im[:, :, np.newaxis] if IMAGE_CHANNEL == 1 else im[:, :]) / 255.)

    @staticmethod
    def _get_input_lens(sequences):
        lengths = np.asarray([len(_) for _ in sequences], dtype=np.int64)
        return sequences, lengths

    def generate_batch_by_files(self, image_index=None):
        batch = {}
        image_batch = []
        label_batch = []

        if image_index:
            # if len(image_index) == TEST_BATCH_SIZE:
            #     ii = image_index[0]
            #     ll = self._label_list[ii]
            #     ll = "".join([GEN_CHAR_SET[_] for _ in ll])
            #     import shutil
            #     shutil.copy(self.image_path[ii], "image/{}.png".format(ll))
            for i, index in enumerate(image_index):
                try:
                    is_training = len(image_index) == BATCH_SIZE
                    is_random = bool(random.getrandbits(1))

                    image_array = self._image(self.image_path[index], is_random=is_training and is_random)
                    label_array = self._label_list[index]
                    if MULTI_SHAPE:
                        image_shape = "{}x{}".format(image_array.shape[0], image_array.shape[1])
                        if image_shape in batch:
                            batch[image_shape].append((image_array, label_array))
                        else:
                            batch[image_shape] = [(image_array, label_array)]
                    else:
                        image_batch.append(image_array)
                        label_batch.append(label_array)
                except OSError:
                    continue
        # else:
        #     for i, path in enumerate(self.image_path):
        #         try:
        #             if i == 0:
        #                 import shutil
        #                 print('----')
        #
        #                 shutil.copy(self.image_path[path], "{}.png".format(self._label_list[path]))
        #             is_random = bool(random.getrandbits(1))
        #             image_array = self._image(self.image_path[path], is_random=is_random)
        #             label_array = self._label_list[path]
        #             if MULTI_SHAPE:
        #                 image_shape = "{}x{}".format(image_array.shape[0], image_array.shape[1])
        #                 if image_shape in batch:
        #                     batch[image_shape].append((image_array, label_array))
        #                 else:
        #                     batch[image_shape] = [(image_array, label_array)]
        #             else:
        #                 image_batch.append(image_array)
        #                 label_batch.append(label_array)
        #         except OSError:
        #             continue

        if MULTI_SHAPE:
            self.label_list = sum([v for k, v in batch.items()], [])
            self.label_list = [i[1] for i in self.label_list]
            return self.classified_generate_batch(batch)
        else:
            if RESIZE[0] == -1:
                image_batch = keras.preprocessing.sequence.pad_sequences(
                    sequences=image_batch,
                    maxlen=None,
                    dtype='float32',
                    padding='post',
                    truncating='post',
                    value=0
                )
                # image_batch = self.padding(image_batch)
            self.label_list = label_batch
            return self.padded_generate_batch(image_batch, label_batch)

    def padded_generate_batch(self, image_batch, label_batch):
        classified_batch = {}
        batch_inputs, batch_seq_len = self._get_input_lens(np.array(image_batch))
        batch_labels = sparse_tuple_from_label(label_batch)
        classified_batch['{}x{}'.format(RESIZE[0], RESIZE[1])] = [batch_inputs, batch_seq_len, batch_labels]
        return classified_batch

    def classified_generate_batch(self, batch):
        classified_batch = {}
        for shape, v in batch.items():
            batch_inputs, batch_seq_len = self._get_input_lens(np.array([i[0] for i in v]))
            batch_labels = sparse_tuple_from_label([i[1] for i in v])
            if shape in classified_batch:
                classified_batch[shape].append([batch_inputs, batch_seq_len, batch_labels])
            else:
                classified_batch[shape] = [batch_inputs, batch_seq_len, batch_labels]
        return classified_batch

    @staticmethod
    def padding(image_batch):

        max_width = max([np.shape(_)[0] for _ in image_batch])
        padded_image_batch = []
        for image in image_batch:
            output_img = np.zeros([max_width, RESIZE[1], IMAGE_CHANNEL])
            output_img[0: np.shape(image)[0]] = image
            padded_image_batch.append(output_img)
        return padded_image_batch

    def generate_batch_by_tfrecords(self, sess):

        _image, _label = sess.run(self.next_element)
        batch = {}
        image_batch = []
        label_batch = []

        for index, (i1, i2) in enumerate(zip(_image, _label)):
            try:
                is_random = bool(random.getrandbits(1))
                random_and_training = is_random and self.mode == RunMode.Trains
                image_array = self._image(i1, is_random=random_and_training)
                label_array = self._encoder(i2)
                if MULTI_SHAPE:
                    image_shape = "{}x{}".format(image_array.shape[0], image_array.shape[1])
                    if image_shape in batch:
                        batch[image_shape].append((image_array, label_array))
                    else:
                        batch[image_shape] = [(image_array, label_array)]
                else:
                    image_batch.append(image_array)
                    label_batch.append(label_array)

            except OSError:
                continue

        if MULTI_SHAPE:
            self.label_list = sum([v for k, v in batch.items()], [])
            self.label_list = [i[1] for i in self.label_list]
            return self.classified_generate_batch(batch)
        else:
            if RESIZE[0] == -1:
                # image_batch = self.padding(image_batch)
                image_batch = keras.preprocessing.sequence.pad_sequences(
                    sequences=image_batch,
                    maxlen=None,
                    dtype='float32',
                    padding='post',
                    truncating='post',
                    value=0
                )
            self.label_list = label_batch
            return self.padded_generate_batch(image_batch, label_batch)


def accuracy_calculation(original_seq, decoded_seq, ignore_value=None):
    if ignore_value is None:
        ignore_value = [-1]
    original_seq_len = len(original_seq)
    decoded_seq_len = len(decoded_seq)
    if original_seq_len != decoded_seq_len:
        tf.logging.error(original_seq)
        tf.logging.error('original lengths {} is different from the decoded_seq {}, please check again'.format(
            original_seq_len,
            decoded_seq_len
        ))
        return 0
    count = 0
    # Here is for debugging, positioning error source use
    error_sample = []
    for i, origin_label in enumerate(original_seq):
        decoded_label = [j for j in decoded_seq[i] if j not in ignore_value]
        if i < 5:
            tf.logging.info(
                "{} {} {} {} {} --> {} {}".format(
                    i,
                    len(origin_label),
                    len(decoded_label),
                    origin_label,
                    decoded_label,
                    [GEN_CHAR_SET[_] for _ in origin_label],
                    [GEN_CHAR_SET[_] for _ in decoded_label]
                )
            )
        if origin_label == decoded_label:
            count += 1
    # Training is not useful for decoding
    # Here is for debugging, positioning error source use
        if origin_label != decoded_label and len(error_sample) < 5:
            error_sample.append({
                "origin": "".join([GEN_CHAR_SET[_] for _ in origin_label]),
                "decode": "".join([GEN_CHAR_SET[_] for _ in decoded_label])
            })
    tf.logging.error(error_sample)
    return count * 1.0 / len(original_seq)


def sparse_tuple_from_label(sequences, dtype=np.int32):
    indices = []
    values = []
    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(0, len(seq), 1)))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
    return indices, values, shape
