
# -*- coding:utf-8 -*-
from all.models.get_model import get_model
import numpy as np
import result_value.value as gl
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



def caculate_time(tau_actual,time_all_local_all,time_global_aggregation_all):
    it_each_local = max(0.00000001, time_all_local_all / tau_actual)
    it_each_global = time_global_aggregation_all
    return it_each_local,it_each_global

def get_tmp_time_for_executing_remaining(use_min_loss,total_time_recomputed,it_each_local,tau_new,it_each_global):
    if use_min_loss:
        tmp_time_for_executing_remaining = total_time_recomputed + it_each_local * (tau_new + 1) + it_each_global * 2
    else:
       tmp_time_for_executing_remaining = total_time_recomputed + it_each_local * tau_new + it_each_global
    return tmp_time_for_executing_remaining

def get_tau_config(tmp_time_for_executing_remaining,max_time,tau_new,use_min_loss,total_time_recomputed,it_each_global,it_each_local):
    is_last_round_tmp=False
    if tmp_time_for_executing_remaining < max_time:
        tau_config = tau_new
    else:
        if use_min_loss:  # Take into account the additional communication round in the end
            tau_config = int((max_time - total_time_recomputed - 2 * it_each_global - it_each_local) / it_each_local)
        else:
            tau_config = int((max_time - total_time_recomputed - it_each_global) / it_each_local)

        if tau_config < 1:
            tau_config = 1
        elif tau_config > tau_new:
            tau_config = tau_new
        is_last_round_tmp = True
    return tau_config,is_last_round_tmp


def get_final_w(use_min_loss,w_global_min_loss,w_global):
    if use_min_loss:
        w_eval = w_global_min_loss
    else:
        w_eval = w_global
    return w_eval



def init_paramas(w_global_init,tau_setup):
    '''

    :param w_global_init:
    :param tau_setup:
    :return:
    w_global_min_loss:最小损失对应的全局模型参数
    w_global：当前全局模型参数
    loss_min：最小损失
    prev_loss_is_min：标记t-1时刻的loss是否为最小损失
    is_adapt_local:本地是否进行自适应
    tau_config:本地更新频率
    '''
    # 设定全局模型参数为初始值
    w_global = w_global_init
    # 全局最小的损失值
    w_global_min_loss = None
    loss_min = np.inf
    # 前一次全局的最小损失值
    prev_loss_is_min = False
    # 当本地更新次数小于0的时候进行动态调整，设置本地更新次数为1
    if tau_setup < 0:
        is_adapt_local = True
        tau_config = 1
    else:
        # 否则的话不是动态调整本地更新次数那么按照tau_setup来进行下一次本地聚合
        is_adapt_local = False
        tau_config = tau_setup

    return w_global,w_global_min_loss,loss_min,prev_loss_is_min,is_adapt_local,tau_config

def init_learning_paramas():
    # time_global_aggregation_all开始记录所有的数据
    time_global_aggregation_all = None
    # 总的时间
    total_time = 0  # Actual total time, where use_fixed_averaging_slots has no effect
    # 重新估计的时间
    total_time_recomputed = 0  # Recomputed total time using estimated time for each local and global update,
    # using predefined values when use_fixed_averaging_slots = true
    it_each_local = None
    it_each_global = None
    ##是否是最后一次训练
    is_last_round = False
    is_eval_only = False

    tau_new_resume = None
    return time_global_aggregation_all,total_time,total_time_recomputed,it_each_local,it_each_global,is_last_round,is_eval_only,tau_new_resume

def result_list(tau_config):
    dflist = []
    gl.COM_TIMES = gl.COM_TIMES + 1
    dflist.append(gl.COM_TIMES)
    # print('current tau config:',gl.COM_TIMES, tau_config)
    dflist.append(tau_config)
    return dflist

def init_traning_paramas(dim_w):
    w_global = np.zeros(dim_w)
    loss_last_global = 0.0
    loss_w_prev_min_loss = 0.0
    received_loss_local_w_prev_min_loss = False
    data_size_total = 0
    time_all_local_all = 0
    data_size_local_all = []
    tau_actual = 0
    return tau_actual,w_global,data_size_local_all,loss_last_global,loss_w_prev_min_loss,received_loss_local_w_prev_min_loss,data_size_total,time_all_local_all


def msg_split(msg,tau_actual):
    # ['MSG_WEIGHT_TIME_SIZE_CLIENT_TO_SERVER', w, time_all_local, tau_actual, data_size_local,
    # loss_last_global, loss_w_prev_min_loss]
    w_local = msg[1]  # 本地的模型权重
    time_all_local = msg[2]  # 本地执行的时间消耗
    tau_actual = max(tau_actual, msg[3])  # Take max of tau because we wait for the slowest node
    data_size_local = msg[4]  # 本地的数据量
    loss_local_last_global = msg[5]  # 最近一次的本地模型的损失
    loss_local_w_prev_min_loss = msg[6]  # 最小的本地模型的损失值
    return w_local,time_all_local,tau_actual,data_size_local,loss_local_last_global,loss_local_w_prev_min_loss


def get_use_w_global_prev_due_to_nan(w_global,w_global_prev):
    if True in np.isnan(w_global):
        print('*** w_global is NaN, using previous value')
        w_global = w_global_prev  # If current w_global contains NaN value, use previous w_global
        use_w_global_prev_due_to_nan = True
    else:
        use_w_global_prev_due_to_nan = False

    return w_global,use_w_global_prev_due_to_nan



# def process(use_min_loss,loss_min,w_global_prev,loss_last_global,data_size_total,received_loss_local_w_prev_min_loss):
#     '''
#     loss_last_global:t-1时刻迭代的全局损失
#     received_loss_local_w_prev_min_loss
#     loss_w_prev_min_loss
#     loss_w_prev_min_loss
#     w_global_prev
#     :return:
#     '''



def process_loss(loss_last_global,loss_local_last_global,data_size_local,loss_local_w_prev_min_loss,loss_w_prev_min_loss):
    loss_last_global += loss_local_last_global * data_size_local
    if loss_local_w_prev_min_loss is not None:
        loss_w_prev_min_loss += loss_local_w_prev_min_loss * data_size_local
        received_loss_local_w_prev_min_loss = True
    return loss_last_global,loss_w_prev_min_loss




def init_local_data():
    '''
    loss_local_last_global_list:t-1时刻全局模型对应的本地损失值
    loss_w_prev_min_loss_list：最小损失对应全局模型参数的损失值，本地或全局？
    :return:
    '''
    loss_local_last_global_list=[]
    loss_w_prev_min_loss_list=[]


