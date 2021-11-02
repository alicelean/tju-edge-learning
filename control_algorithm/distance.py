import numpy as np
import scipy.stats
import numpy as np
import cv
def KL_divergence(p,q):
    '''
    KL散度越小，越小越相似
    :param p:np.asarray
    :param q:np.asarray
    :return:
    '''
    return scipy.stats.entropy(p, q, base=2)

def JS_divergence(p,q):
    '''
    JLS散度
    :param p:np.asarray
    :param q:np.asarray
    :return:
    '''
    M=(p+q)/2
    return 0.5*scipy.stats.entropy(p, M, base=2)+0.5*scipy.stats.entropy(q, M, base=2)

def EMD_divergenc(p,q):
    '''
    EMDdistance
    :param p:np.asarray
    :param q:np.asarray
    :return:
    '''
    pp = cv.fromarray(p)
    qq = cv.fromarray(q)
    emd = cv.CalcEMD2(pp, qq, cv.CV_DIST_L2)