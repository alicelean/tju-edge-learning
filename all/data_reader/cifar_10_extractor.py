# -*- coding:utf-8 -*-
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from struct import *
import numpy as np
from all.util.utils import get_one_hot_from_label_index

#CIFAR-10 是一个包含60000张图片的数据集。其中每张照片为32*32的彩色照片，每个像素点包括RGB三个数值，数值范围 0 ~ 255。
#所有照片分属10个不同的类别，分别是 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
#其中五万张图片被划分为训练集，剩下的一万张图片属于测试集。
#data_batch_1 ~ data_batch_5 是划分好的训练数据,每个文件里包含10000张图片，test_batch 是测试集数据，也包含10000张图片。他们的结构是一样的

BYTES_EACH_LABEL = 1
#32*32*3=3072 RGB,文件是10000 * 3072 的二维数组，每一行代表一张图片的像素值。
BYTES_EACH_IMAGE = 3072
BYTES_EACH_SAMPLE = BYTES_EACH_LABEL + BYTES_EACH_IMAGE
#每个文件里包含10000张图片
SAMPLES_EACH_FILE = 10000


def cifar_10_extract_samples(sample_list, is_train=True, file_path=os.path.dirname(__file__)):
    '''
    读取部分样本数据
    :param sample_list:为要读取的图片位置list,[0-5000]是训练集，0-1000是测试集
    :param is_train:
    :param file_path:
    :return:
    '''
    #f_list存放要读取的数据文件path
    f_list = []
    if is_train:
        f_list.append(open(file_path + '/cifar-10-batches-bin/data_batch_1.bin', 'rb'))
        f_list.append(open(file_path + '/cifar-10-batches-bin/data_batch_2.bin', 'rb'))
        f_list.append(open(file_path + '/cifar-10-batches-bin/data_batch_3.bin', 'rb'))
        f_list.append(open(file_path + '/cifar-10-batches-bin/data_batch_4.bin', 'rb'))
        f_list.append(open(file_path + '/cifar-10-batches-bin/data_batch_5.bin', 'rb'))
    else:
        f_list.append(open(file_path + '/cifar-10-batches-bin/test_batch.bin', 'rb'))

    data = []
    labels = []
    # 读取数据,这几个文件都是通过 pickle 产生的，所以在读取的时候也要用到这个包
    for i in sample_list:
        #（0-1000）在第0个文件
        #（1000-2000）在第一个文件
        #（2000-3000）在第二个文件
        file_index = int(i / float(SAMPLES_EACH_FILE))
        #从文件头开始的偏移量offset
        offset=(i - file_index * SAMPLES_EACH_FILE) * BYTES_EACH_SAMPLE
        #文件指针进行偏移到所需要的数据位置
        f_list[file_index].seek(offset)
        label = unpack('>B', f_list[file_index].read(1))[0]

        y = get_one_hot_from_label_index(label)
        x = np.array(list(f_list[file_index].read(BYTES_EACH_IMAGE)))

        # Simple normalization (choose either this or next approach)
        # x = x / 255.0

        # Normalize according to mean and standard deviation as suggested in Tensorflow tutorial
        tmp_mean = np.mean(x)
        tmp_stddev = np.std(x)
        tmp_adjusted_stddev = max(tmp_stddev, 1.0 / np.sqrt(len(x)))
        x = (x - tmp_mean) / tmp_adjusted_stddev

        # Reshaping to match with Tensorflow format
        x = np.reshape(x, [32, 32, 3], order='F')
        x = np.reshape(x, [3072], order='C')

        data.append(x)
        labels.append(y)

    for f in f_list:
        f.close()

    return data, labels


def cifar_10_extract(start_sample_index, num_samples, is_train=True, file_path=os.path.dirname(__file__)):
    '''
    根据起始位置和num来获取sample_list
    :param start_sample_index:
    :param num_samples:
    :param is_train:
    :param file_path:
    :return:
    '''
    sample_list = range(start_sample_index, start_sample_index + num_samples)
    return cifar_10_extract_samples(sample_list, is_train, file_path)