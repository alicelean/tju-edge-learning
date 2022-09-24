import numpy as np
def FedAvg(w_local_all,data_size_local_all,w_global,datalength):
    '''
    聚合算法
    :param w_local_all: 所有的本地模型的模型参数列表
    :param data_size_local_all: 各个节点的本地数据分布规律
    :param w_global: 当前的全局模型
    :return: 新的全局模型
    '''
    for i in range(len(w_local_all)):
        w_local = w_local_all[i]
        data_size_local = data_size_local_all[i]
        rate = float(data_size_local) / float(datalength)
        w_global = w_global + w_local * rate

    return w_global

def DisAg(nodew,w_global,w_local_all):
    for i in range(len(w_local_all)):
        w_local = w_local_all[i]
        w_global = w_global + w_local * nodew[i]
    return w_global

def FedDis(nodew,w_local_all,data_size_local_all,w_global,datalength):
    '''
    聚合算法
    :param w_local_all: 所有的本地模型的模型参数列表
    :param data_size_local_all: 各个节点的本地数据分布规律
    :param w_global: 当前的全局模型
    :return: 新的全局模型
    '''
    for i in range(len(w_local_all)):
        w_local = w_local_all[i]
        data_size_local = data_size_local_all[i]
        rate = float(data_size_local) / float(datalength)
        w_global = w_global + w_local * (rate+nodew[i])

    return w_global

def Fed_Dis(nodew,w_local_all,data_size_local_all,w_global,datalength):
    '''
    聚合算法
    :param w_local_all: 所有的本地模型的模型参数列表
    :param data_size_local_all: 各个节点的本地数据分布规律
    :param w_global: 当前的全局模型
    :return: 新的全局模型
    '''
    for i in range(len(w_local_all)):
        w_local = w_local_all[i]
        data_size_local = data_size_local_all[i]
        rate = float(data_size_local) / float(datalength)
        w_global = w_global + w_local * (rate*nodew[i])

    return w_global


def AM(w_local_all,w_global):
    '''
    平均聚合
    :param w_local_all: 所有的本地模型的模型参数列表
    :param w_global: 当前的全局模型
    :return: 新的全局模型
    '''
    rate=1.0/float(len(w_local_all))
    for i in range(len(w_local_all)):
        w_local = w_local_all[i]
        w_global = w_global + w_local * rate
    return w_global


def Dynamic_Aggr(w_local_all,loss_local_all,w_global):
    '''
    :param w_local_all:
    :param loss_local_all:
    :param w_global:
    :return:
    '''



