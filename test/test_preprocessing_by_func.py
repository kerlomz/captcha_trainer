#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>

if __name__ == '__main__':
    import io
    import os
    import PIL.Image
    import hashlib
    import cv2
    import numpy as np
    from pretreatment import preprocessing_by_func

    src_color = "red"
    root_dir = r"E:\Task\Tax\ChinaTaxTest-01_Trains\{}".format(src_color)
    target_dir = r"H:\Samples\tax_gen\real\{}2red".format(src_color)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for name in os.listdir(root_dir):
        label = name.split("_")[0]
        path = os.path.join(root_dir, name)
        with open(path, "rb") as f:
            path_or_bytes = f.read()
        path_or_stream = io.BytesIO(path_or_bytes)
        pil_image = PIL.Image.open(path_or_stream).convert("RGB")
        im = np.array(pil_image)
        im = preprocessing_by_func(exec_map={
            "black": [
                "$$target_arr[:, :, 2] = 255 - target_arr[:, :, 2]",
            ],
            "red": [],
            "yellow": [
                "@@target_arr[:, :, (0, 0, 1)]",
            ],
            "blue": [
                "@@target_arr[:, :, (1, 2, 0)]",
            ]
        },
            src_arr=im,
            key=src_color
        )
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        cv_img = cv2.imencode('.png', im)[1]
        img_bytes = bytes(bytearray(cv_img))
        tag = hashlib.md5(img_bytes).hexdigest()
        new_name = "{}_{}.png".format(label, tag)
        new_path = os.path.join(target_dir, new_name)
        print(new_name)
        with open(new_path, "wb") as f:
            f.write(img_bytes)