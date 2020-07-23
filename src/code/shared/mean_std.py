"""
in this script, we calculate the image per channel mean and standard
deviation in the training set, do not calculate the statistics on the
whole dataset, as per here http://cs231n.github.io/neural-networks-2/#datapre
"""

import numpy as np
from os import listdir
from os.path import join, isdir
from glob import glob
import cv2
import timeit

# number of channels of the dataset image, 3 for color jpg, 1 for grayscale img
# you need to change it to reflect your dataset
CHANNEL_NUM = 3
global_path='/nfs/students/summer-term-2020/project-3/src'


def cal_stat_downsample(image_txt,sample_rate):
    pixel_num = 0  # store all pixel number in the dataset
    channel_sum = np.zeros(CHANNEL_NUM)
    channel_sum_squared = np.zeros(CHANNEL_NUM)
    with open(image_txt,'r') as txt:
        img_paths=txt.read().splitlines()
        for idx, img_path in enumerate(img_paths):
            if idx%sample_rate==0:
                path=join(global_path,img_path)
                im = cv2.imread(path)  # image in M*N*CHANNEL_NUM shape, channel in BGR order
                im = im / 255.0
                pixel_num += (im.size / CHANNEL_NUM)
                channel_sum += np.sum(im, axis=(0, 1))
                channel_sum_squared += np.sum(np.square(im), axis=(0, 1))
        bgr_mean = channel_sum / pixel_num
        bgr_std = np.sqrt(channel_sum_squared / pixel_num - np.square(bgr_mean))
        rgb_mean = list(bgr_mean)[::-1]
        rgb_std = list(bgr_std)[::-1]
        return rgb_mean,rgb_std

def cal_stat_all(image_txt):
    pixel_num = 0  # store all pixel number in the dataset
    channel_sum = np.zeros(CHANNEL_NUM)
    channel_sum_squared = np.zeros(CHANNEL_NUM)
    with open(image_txt,'r') as txt:
        img_paths=txt.read().splitlines()
        for img_path in img_paths:
            path=join(global_path,img_path)
            im = cv2.imread(path)  # image in M*N*CHANNEL_NUM shape, channel in BGR order
            im = im / 255.0
            pixel_num += (im.size / CHANNEL_NUM)
            channel_sum += np.sum(im, axis=(0, 1))
            channel_sum_squared += np.sum(np.square(im), axis=(0, 1))
        bgr_mean = channel_sum / pixel_num
        bgr_std = np.sqrt(channel_sum_squared / pixel_num - np.square(bgr_mean))
        rgb_mean = list(bgr_mean)[::-1]
        rgb_std = list(bgr_std)[::-1]
        return rgb_mean,rgb_std
train_txt='/nfs/students/summer-term-2020/project-3/src/data2/daytime_train.txt'
#downsample=100
start = timeit.default_timer()
mean, std = cal_stat_all(train_txt)
end = timeit.default_timer()
print("elapsed time: {}".format(end - start))
print("mean:{}\nstd:{}".format(mean, std))
