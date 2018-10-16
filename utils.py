#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import numpy as np
import PIL.Image
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
        self.data_dir = PATH_MAP[mode]
        self.image = []
        self.image_path = []
        self.label_list = []
        for root, sub_folder, file_list in os.walk(self.data_dir):
            for file_path in file_list:
                image_name = os.path.join(root, file_path)
                self.image_path.append(image_name)
                # Get the label from the file name based on the regular expression.
                code = re.search(
                    REGEX_MAP[mode], image_name.split(PATH_SPLIT)[-1]
                ).group()
                # The manual verification code platform is not case sensitive,
                # - it will affect the accuracy of the training set.
                # Here is a case conversion based on the selected character set.
                code = code.lower() if 'LOWER' in CHAR_SET else code
                code = code.upper() if 'UPPER' in CHAR_SET else code
                code = [encode_maps()[c] for c in list(code)]
                self.label_list.append(code)

    @property
    def size(self):
        return len(self.label_list)

    def the_label(self, index_list):
        labels = []
        for i in index_list:
            labels.append(self.label_list[i])

        return labels

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

    def input_index_generate_batch(self, index):

        image_batch = [self._image(self.image_path[i]) for i in index]
        label_batch = [self.label_list[i] for i in index]

        def get_input_lens(sequences):
            # OUT_CHANNEL is the output channels of the last layer of CNN
            lengths = np.asarray([OUT_CHANNEL for _ in sequences], dtype=np.int64)
            return sequences, lengths

        batch_inputs, batch_seq_len = get_input_lens(np.array(image_batch))
        batch_labels = sparse_tuple_from_label(label_batch)

        return batch_inputs, batch_seq_len, batch_labels


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
