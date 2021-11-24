import numpy as np
from scipy.spatial.distance import pdist

#方法一：根据公式求解
def Euc_distance(x,y):
    '''
    欧式距离
    :param x:
    :param y:
    :return:
    '''
    d1=np.sqrt(np.sum(np.square(x-y)))
    return d1


def Sde_distance(x,y):
    '''
    标准化欧氏距离
    :param x:
    :param y:
    :return:
    '''
    X = np.vstack([x, y])
    sk = np.var(X, axis=0, ddof=1)
    d1 = np.sqrt(((x - y) ** 2 / sk).sum())
    return d1

def Manh_distance(x,y):
    '''
    麦哈顿距离
    :param x:
    :param y:
    :return:
    '''
    d1=np.sum(np.abs(x-y))
    return d1
def Mab_distance(x,y):
    '''
    马氏距离
    :param x
    :param y
    :return:
    '''
    # 马氏距离要求样本数要大于维数，否则无法求协方差矩阵
    # 此处进行转置，表示10个样本，每个样本2维
    X = np.vstack([x, y])
    XT = X.T
    S = np.cov(X)  # 两个维度之间协方差矩阵
    SI = np.linalg.inv(S)  # 协方差矩阵的逆矩阵
    # 马氏距离计算两个样本之间的距离，此处共有10个样本，两两组合，共有45个距离。
    n = XT.shape[0]
    d1 = []
    for i in range(0, n):
        for j in range(i + 1, n):
            delta = XT[i] - XT[j]
            d = np.sqrt(np.dot(np.dot(delta, SI), delta.T))
            d1.append(d)
    return d1

def Chb_distance(x, y):
    '''
    切比雪夫距离
    :param x:
    :param y:
    :return:
    '''
    d1=np.max(np.abs(x-y))
    return d1


def Mik_distance(x, y):
    '''
    闵可夫斯基距离
    :param x:
    :param y:
    :return:
    '''
    X = np.vstack([x, y])
    d1= pdist(X, 'minkowski', p=2)
    return d1[0]

def alldistance(y_list,Y,dis_type=None):
    '''
    根据dis_type进行不同的距离计算
    :param y_list: list
    :param Y: list
    :param dis_type:
    :return: dis:距离值
    '''
    if dis_type is not None:
        if dis_type == 'Euc':
            dis = Euc_distance(y_list, Y)
        elif dis_type == 'Sde':
            dis = Sde_distance(y_list, Y)
        elif dis_type == 'Manh':
            dis = Manh_distance(y_list, Y)
        elif dis_type == 'Chb':
            dis = Chb_distance(y_list, Y)
        elif dis_type == 'Mik':
            dis = Mik_distance(y_list, Y)
        elif dis_type == 'Mab':
            dis = Mab_distance(y_list, Y)
    else:
        dis = Euc_distance(y_list, Y)
    return dis