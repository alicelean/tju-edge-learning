import socket
import time
from data_reader.data_reader import *
import result_value.value as gl
import pandas as pd
from util.utils import *
from util.tools import *
from config import *
from plot.methods import *
model=createmodel(model_name,step_size)
n_nodes=5
#建立网络通信
listening_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listening_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listening_sock.bind((SERVER_ADDR, SERVER_PORT))
client_sock_all=[]
print("----------------------------------------------------waiting client connection-----------------------------------------------")
# 建立多个网络链接
while len(client_sock_all) < n_nodes:
    listening_sock.listen(5)
    (client_sock, (ip, port)) = listening_sock.accept()
    print(len(client_sock_all),'-------------------Got connection from ', (ip,port),"--------------------------")
    client_sock_all.append(client_sock)

print("----------------------------------------------------server reading data-----------------------------------------------")

#读取数据
train_image, train_label, test_image, test_label, train_label_orig = get_minist_data(dataset, total_data,dataset_file_path)
indices_each_node= get_case_1(n_nodes,train_label_orig)
df = pd.DataFrame(columns=['sim','case','loss','accracy'])
#本地更新运行频率
#tau_list=[3,5,10,20,30,40,50,80,100,120,180,220,260,320]
tau_list=[600]
#total_time,5,10,30,50,100
total_time=50
minibatch=3

for tau in tau_list:
    #统计对象的初始化,对模型的权重进行初始化
    dim_w = model.get_weight_dimension(train_image, train_label)
    w_global_init = model.get_init_weight(dim_w, rand_seed=0)
    w_global, w_global_min_loss, loss_min, prev_loss_is_min, is_adapt_local, tau_config=init_paramas(w_global_init,tau)
    #每个节点都要进行处理,送初始数据
    for node_i in range(0, n_nodes):
        indices_this_node=indices_each_node[node_i]
        msg =['MSG_INIT_SERVER_TO_CLIENT', model_name, dataset,minibatch, step_size,
                       batch_size, total_data,indices_this_node ,
                       use_min_loss,node_i]
        send_msg(client_sock_all[node_i], msg)
        print(time.time()," node ",node_i,"MSG_INIT_SERVER_TO_CLIENT msg has send")
    print('-----------------------------------------All clients connected，init msg  has been sended-------------------------------------------------')

    #接收消息，知道各个节点都已经收到了相对应的数据
    for node_i in range(0, n_nodes):
        recv_msg(client_sock_all[node_i], 'MSG_DATA_PREP_FINISHED_CLIENT_TO_SERVER')
        #开始进行边缘节点的协作训练
        print("node_i :",node_i,'------------------------------------------------Ready to training----------------------------------------------------------')
    learning_time=0
    is_last_round=False
    prev_loss_is_min=False
    w_global_prev=w_global
    iter_times=0
    actual_times=0
    time_total_all_start = time.time()
    last_is_nan=False
    while True:
        iter_times=iter_times+1
        #server 数据全局模型数据传送给client
        for node_i in range(0, n_nodes):
            msg = ['MSG_WEIGHT_TAU_SERVER_TO_CLIENT', w_global, tau_config, is_last_round, prev_loss_is_min,node_i,iter_times]
            send_msg(client_sock_all[node_i], msg)
            print(time.time(),"  ",iter_times," local iteration node ", node_i, "global msg has send")
        if is_last_round:
            break
        else:
            # 记录t-1时刻全局模型参数值
            if not last_is_nan:
                w_global_prev = w_global
            max_local_time = 0
            w_local_all = []
            data_size_local_all = []
            grad_list=[]
            datalength = 0
            print("..................server msg have send , Waiting for the ", iter_times,
                  " local iteration at client..................................")
            print("server is_last_round", is_last_round)
            for node_i in range(0, n_nodes):
                msg = recv_msg(client_sock_all[node_i], 'MSG_WEIGHT_TIME_SIZE_CLIENT_TO_SERVER')
                #['MSG_WEIGHT_TIME_SIZE_CLIENT_TO_SERVER', w, time_local, tau_actual, data_size_local]
                if msg[2]>max_local_time:
                    max_local_time=msg[2]
                w_local_all.append(msg[1])
                datalength+=msg[4]
                data_size_local_all.append(msg[4])
                #记录每个边缘节点的梯度
                grad_list.append(msg[5])
            print("..................", iter_times," local iteration at client has finished..................................")

            for i  in range(len(w_local_all)):
                w_local=w_local_all[i]
                data_size_local=data_size_local_all[i]
                rate=float(data_size_local)/float(datalength)
                w_global=w_global+w_local*rate
                print(w_local)
            if True in np.isnan(w_global):
                w_global = w_global_prev
                last_is_nan = True
            actual_times=actual_times+max_local_time
            if actual_times>total_time:
                is_last_round=True
    w_eval=w_global
    loss_final = model.loss(train_image, train_label, w_eval)
    accuracy_final = model.accuracy(test_image, test_label, w_eval)
    redf = pd.DataFrame(columns=["method","n_nodes","total_time","minibatch","iter_times", "tau", "loss", "accuracy"])
    redf.loc[len(redf)+1]=["FedAvg",n_nodes,total_time,minibatch,iter_times,tau,loss_final,accuracy_final]
    redf.to_csv(gl.PATH+'case_1.csv', mode='a', header=False)



#             time_global_aggregation_all, total_time, total_time_recomputed, it_each_local, it_each_global, is_last_round, is_eval_only, tau_new_resume=init_learning_paramas()
#             ###正式开始进行训练---------------------------------------------------
#             while True:
#                 # 当前运行中的数据存储，方便对实验结果进行分析
#                 time_total_all_start = time.time()
#                 dflist=result_list(tau_config)
#
#                 #server 数据全局模型数据传送给client
#                 for n in range(0, n_nodes):
#                     msg = ['MSG_WEIGHT_TAU_SERVER_TO_CLIENT', w_global, tau_config, is_last_round, prev_loss_is_min]
#                     send_msg(client_sock_all[n], msg)
#                 #记录t-1时刻全局模型参数值
#                 w_global_prev = w_global
#
#                 tau_actual,w_global,data_size_local_all,loss_last_global,loss_w_prev_min_loss,received_loss_local_w_prev_min_loss,data_size_total,time_all_local_all=init_traning_paramas(dim_w)
#                 #对每一个节点进行操作，当一轮本地更新结束后，会进行数据搜集，并聚合
#
#                 for n in range(0, n_nodes):
#                     msg = recv_msg(client_sock_all[n], 'MSG_WEIGHT_TIME_SIZE_CLIENT_TO_SERVER')
#                     w_local, time_all_local, tau_actual, data_size_local, loss_local_last_global, loss_local_w_prev_min_loss=msg_split(msg,tau_actual)
#                     time_all_local_all = max(time_all_local_all, time_all_local)   #Use max. time to take into account the slowest node
#                     data_size_local_all.append(data_size_local)
#                     data_size_total += data_size_local
#
#                     # 参数聚合
#                     w_global += w_local * data_size_local
#
#                     loss_last_global += loss_local_last_global * data_size_local
#                     if loss_local_w_prev_min_loss is not None:
#                         loss_w_prev_min_loss += loss_local_w_prev_min_loss * data_size_local
#                         received_loss_local_w_prev_min_loss = True
#
#
#                     # if use_min_loss:
#                     #     #计算最近一次全局损失（F(w)=all(Fi(w)*data_size_local/data_size_total）)
#                     #     loss_last_global += loss_local_last_global * data_size_local
#                     #     #计算目前的最小损失
#                     #     if loss_local_w_prev_min_loss is not None:
#                     #         loss_w_prev_min_loss += loss_local_w_prev_min_loss * data_size_local
#                     #         received_loss_local_w_prev_min_loss = True
#
#                 w_global /= data_size_total
#                 loss_last_global /= data_size_total
# #此时的w_global已经是一个全局模型的权重参数
# #
#                 w_global, use_w_global_prev_due_to_nan = get_use_w_global_prev_due_to_nan(w_global, w_global_prev)
#
#                 if received_loss_local_w_prev_min_loss:
#                     loss_w_prev_min_loss /= data_size_total
#                     loss_min = loss_w_prev_min_loss
#
#                 if loss_last_global < loss_min:
#                     loss_min = loss_last_global
#                     w_global_min_loss = w_global_prev
#                     prev_loss_is_min = True
#                 else:
#                     prev_loss_is_min = False
#
#                     # print("Loss of previous global value: " + str(loss_last_global))
#                     # print("Minimum loss: " + str(loss_min))
#
#                 # If use_w_global_prev_due_to_nan, then use tau = 1 for next round
#                 if not use_w_global_prev_due_to_nan:
#                         if tau_new_resume is not None:
#                             tau_new = tau_new_resume
#                             tau_new_resume = None
#                         else:
#                             tau_new = tau_config
#                 else:
#                     if tau_new_resume is None:
#                         tau_new_resume = tau_config
#                     tau_new = 1
#
#                 # Calculate time,计算资源消耗
#                 time_total_all_end = time.time()
#                 time_total_all = time_total_all_end - time_total_all_start
#                 time_global_aggregation_all = max(0.0, time_total_all - time_all_local_all)
#                 local_time=time_all_local_all / tau_actual
#                 print('Time for one local iteration:', local_time)
#                 print('Time for global averaging:', time_global_aggregation_all)
#
#                 #参数存放
#                 dflist.append(time_total_all)
#                 dflist.append(local_time)
#                 dflist.append(time_global_aggregation_all)
#                 dflist.append(loss_last_global)
#                 gl.DF.loc[len(gl.DF)+ 1] = dflist
#
#                 #计算it_each_local，it_each_global
#                 it_each_local,it_each_global=caculate_time( tau_actual, time_all_local_all,
#                               time_global_aggregation_all)
#
#                 #Compute number of iterations is current slot，Compute time in current slot
#                 total_time_recomputed += it_each_local * tau_actual + it_each_global
#                 total_time += time_total_all
#
#                 # #stat.collect_stat_end_local_round(case, tau_actual, it_each_local, it_each_global, control_alg, model,
#                 #                                   train_image, train_label, test_image, test_label, w_global,
#                 #                                   total_time_recomputed)
#
#                 # Check remaining resource budget (use a smaller tau if the remaining time is not sufficient)
#                 # is_last_round_tmp = False
#                 tmp_time_for_executing_remaining=get_tmp_time_for_executing_remaining(use_min_loss,total_time_recomputed,it_each_local,tau_new,it_each_global)
#                 tau_config,is_last_round_tmp=get_tau_config(tmp_time_for_executing_remaining, max_time, tau_new, use_min_loss,
#                                    total_time_recomputed, it_each_global, it_each_local)
#
#                 if is_last_round:
#                     break
#                 if is_eval_only:
#                     tau_config = 1
#                     is_last_round = True
#                 if is_last_round_tmp:
#                     if use_min_loss:
#                         is_eval_only = True
#                     else:
#                         is_last_round = True
#
#             #w_eval 是最终的模型权重数据，也就是server上的模型
#             # w_global_min_loss，最小损失对应的模型权重
#             w_eval =get_final_w(use_min_loss, w_global_min_loss, w_global)
#
#             #stat.collect_stat_end_global_round(sim, case, tau_setup, total_time, model, train_image, train_label,
#             #                                    test_image, test_label, w_eval, total_time_recomputed)
#             loss_final = model.loss(train_image, train_label, w_eval)
#             accuracy_final = model.accuracy(test_image, test_label, w_eval)
#             print(sim,case,loss_final,accuracy_final)
#             dl = [sim,case,loss_final,accuracy_final]
#             df.loc[len(df) + 1]=dl
#     df.to_csv(gl.PATH+"accuracy.csv")




