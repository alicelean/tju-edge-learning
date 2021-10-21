# -*- coding:utf-8 -*-
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from all.util.utils import get_index_from_one_hot_label,get_even_odd_from_one_hot_label
from all.data_reader.mnist_extractor import mnist_extract
from all.data_reader.mnist_extractor import mnist_extract_samples
from all.data_reader.cifar_10_extractor import cifar_10_extract
from all.data_reader.cifar_10_extractor import cifar_10_extract_samples





def get_data(dataset, total_data, dataset_file_path=os.path.dirname(__file__), sim_round=None):
    '''
    读取文件数据
    :param dataset: MNIST_ORIG_EVEN_ODD（奇偶数据），MNIST_ORIG_ALL_LABELS（0-9数据），
    :param total_data:
    :param dataset_file_path:
    :param sim_round:
    :return: train_image, train_label, test_image, test_label, train_label_orig
    '''
    if dataset == 'MNIST_ORIG_EVEN_ODD' or dataset == 'MNIST_ORIG_ALL_LABELS':

        #total_data_train训练数据总量，total_data_test测试数据总量
        if total_data > 60000:
            total_data_train = 60000
        else:
            total_data_train = total_data

        if total_data > 10000:
            total_data_test = 10000
        else:
            total_data_test = total_data


        #sim_round随机种子，默认为none
        start_index_train = 0
        start_index_test = 0

        if sim_round is not None:
            start_index_train = (sim_round * total_data_train) % (max(1, 60000 - total_data_train + 1))
            start_index_test = (sim_round * total_data_test) % (max(1, 10000 - total_data_test + 1))
            print("sim is ",sim_round,"start_index_train is ",start_index_train,"start_index_test is ",start_index_test)

        #从文件中获取数据
        train_image, train_label = mnist_extract(start_index_train, total_data_train, True, dataset_file_path)
        test_image, test_label = mnist_extract(start_index_test, total_data_test, False, dataset_file_path)

        # train_label_orig must be determined before the values in train_label are overwritten below
        #train_label_orig=[1,3,0,2,.....]原始类别标签列表
        train_label_orig=[]
        for i in range(0, len(train_label)):
            label = get_index_from_one_hot_label(train_label[i])
            train_label_orig.append(label[0])

        if dataset == 'MNIST_ORIG_EVEN_ODD':
            for i in range(0, len(train_label)):
                train_label[i] = get_even_odd_from_one_hot_label(train_label[i])

            for i in range(0, len(test_label)):
                test_label[i] = get_even_odd_from_one_hot_label(test_label[i])

    elif dataset == 'CIFAR_10':

        if total_data > 50000:
            total_data_train = 50000
        else:
            total_data_train = total_data

        if total_data > 10000:
            total_data_test = 10000
        else:
            total_data_test = total_data

        train_image, train_label = cifar_10_extract(0, total_data_train, True, dataset_file_path)
        test_image, test_label = cifar_10_extract(0, total_data_test, False, dataset_file_path)

        train_label_orig=[]
        for i in range(0, len(train_label)):
            label = get_index_from_one_hot_label(train_label[i])
            train_label_orig.append(label[0])

    else:
        raise Exception('Unknown dataset name.')

    return train_image, train_label, test_image, test_label, train_label_orig


def get_data_train_samples(dataset, samples_list, dataset_file_path=os.path.dirname(__file__)):
    '''
    获取部分的样本数据train_image, train_label
    :param dataset:
    :param samples_list:
    :param dataset_file_path:
    :return: train_image, train_label
    '''
    if dataset == 'MNIST_ORIG_EVEN_ODD' or dataset == 'MNIST_ORIG_ALL_LABELS':
        train_image, train_label = mnist_extract_samples(samples_list, True, dataset_file_path)
        if dataset == 'MNIST_ORIG_EVEN_ODD':
            for i in range(0, len(train_label)):
                train_label[i] = get_even_odd_from_one_hot_label(train_label[i])

    elif dataset == 'CIFAR_10':
        train_image, train_label = cifar_10_extract_samples(samples_list, True, dataset_file_path)

    else:
        raise Exception('Training data sampling not supported for the given dataset name, use entire dataset by setting batch_size = total_data, ' +
                        'also confirm that dataset name is correct.')

    return train_image, train_label
