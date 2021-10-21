
# -*- coding:utf-8 -*-
from all.models.get_model import get_model

def createmodel(model_name,step_size):
    '''
    创建模型
    :param model_name: 当前的模型，不同的名称对应不同的模型
    :return: model
    '''
    model = get_model(model_name)
    ##神经网络需要创建计算流图
    if hasattr(model, 'create_graph'):
        model.create_graph(learning_rate=step_size)
    return model