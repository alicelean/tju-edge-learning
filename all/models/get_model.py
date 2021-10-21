# -*- coding:utf-8 -*-
import os, sys
from all.models.cnn_mnist import ModelCNNMnist
from all.models.cnn_cifar10 import ModelCNNCifar10
from all.models.svm_smooth import ModelSVMSmooth

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def get_model(model_class_name):
    '''
    构建不同的model
    :param model_class_name: 模型名字
    :return: model

    '''
    model=None
    if model_class_name == 'ModelCNNMnist':
        model=ModelCNNMnist()
    elif model_class_name == 'ModelCNNCifar10':
        model=ModelCNNCifar10()
    elif model_class_name == 'ModelSVMSmooth':
        model=ModelSVMSmooth()
    else:
        raise Exception("Unknown model class name")

    return model
