
# -*- coding:utf-8 -*-
from all.models.get_model import get_model
import numpy as np
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



def caculate_time(case,time_gen,use_fixed_averaging_slots,tau_actual,time_all_local_all,time_global_aggregation_all):
    if use_fixed_averaging_slots:
        if isinstance(time_gen, (list,)):
            t_g = time_gen[case]
        else:
            t_g = time_gen
        it_each_local = max(0.00000001, np.sum(t_g.get_local(tau_actual)) / tau_actual)
        it_each_global = t_g.get_global(1)[0]
    else:
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