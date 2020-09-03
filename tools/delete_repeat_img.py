# -*- coding: utf-8 -*-

import os
import cv2 as cv
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from skimage.measure import compare_ssim

EXT = ['.jpg', '.jpeg']
path = r'C:\Users\sml2h\Desktop\sb'
codes = [item.split("_")[0].lower() for item in os.listdir(path)]
codes = list(set(codes))
codes_dict = {}

for code in codes:
    codes_dict[code] = []
for item in os.listdir(path):
    codes_dict[item.split("_")[0].lower()].append(item)

def delete(imgs_n):
    # return
    for image in imgs_n:
        os.remove(image)

#
def find_sim_images(code_lists):
    imgs_n = []
    img_files = [os.path.join(path, code) for code in code_lists]
    for currIndex, filename in enumerate(img_files):
        if filename in imgs_n:
            continue
        if currIndex >= len(img_files) - 1:
            break
        for filename2 in img_files[currIndex + 1:]:
            if filename2 in imgs_n:
                continue
            img = cv.imdecode(np.fromfile(filename, dtype=np.uint8), -1)
            img1 = cv.imdecode(np.fromfile(filename2, dtype=np.uint8), -1)
            try:
                ssim = compare_ssim(img, img1, multichannel=True)
                if ssim > 0.9:
                    imgs_n.append(filename2)
                    print(filename, filename2, ssim)
            except ValueError:
                pass
    print(imgs_n)
    delete(imgs_n)
    return imgs_n


with ThreadPoolExecutor(max_workers=30) as t:
    tasks = []
    for key in codes_dict:
        codes = codes_dict[key]
        task = t.submit(find_sim_images, codes)
        tasks.append(task)
    while True:
        result = []
        for i in range(len(tasks)):
            result.append(task.done())
        if False not in result:
            break
        time.sleep(1)