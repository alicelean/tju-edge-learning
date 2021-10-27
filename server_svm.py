import socket
import time
from data_reader.data_reader import *
import result_value.value as gl
import pandas as pd
from util.utils import *
from util.tools import *
from config import *
from plot.methods import *
model = createmodel(model_name, step_size)
n_nodes = 5
# 建立网络通信
listening_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listening_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listening_sock.bind((SERVER_ADDR, SERVER_PORT))
client_sock_all = []
# 建立多个网络链接
print("---------------------------------------- waiting connetion-----------------------------------------------")
while len(client_sock_all) < n_nodes:
    listening_sock.listen(5)
    (client_sock, (ip, port)) = listening_sock.accept()
    print(len(client_sock_all), '-------------------Got connection from ', (ip, port),"--------------------------")
    client_sock_all.append(client_sock)
print("----------------------------------------------------waiting client connection-----------------------------------------------")
print("----------------------------------------------------server reading data-----------------------------------------------")
# 读取数据
train_image, train_label, test_image, test_label, train_label_orig = get_minist_data(dataset, total_data,
                                                                                     dataset_file_path)
indices_each_node = get_case_1(n_nodes, train_label_orig)
# tau_list=[3,5,10,20,30,40,50,80,100,120,180,220,260,320]
# total_time,5,10,30,50,100
time_list=[]
for i in range(10, 1000, 10):
    time_list.append(i)
minibatch = 3
for t_time in time_list:
    total_time = t_time
    tau_list =[]
    for i in range (5,1005,5):
        tau_list.append(i)
    print (tau_list)
    for t in range(len(tau_list)):
        tau=tau_list[t]
        # 本地更新运行频率
        # 统计对象的初始化,对模型的权重进行初始化
        print("------------------------------------------------start" ,str(t), "experiments-------------------------------------------------------------")
        dim_w = model.get_weight_dimension(train_image, train_label)
        w_global = model.get_init_weight(dim_w, rand_seed=0)
        w_global_prev = w_global
        tau_config=tau
        learning_time = 0
        is_last_round = False
        #迭代次数
        iter_times = 0
        #训练实际用的时间
        actual_times = 0
        #对nan进行特殊处理
        last_is_nan = False
        prev_loss_is_min=False
        # 每个节点都要进行处理,送初始数据
        for node_i in range(0, n_nodes):
            indices_this_node = indices_each_node[node_i]
            msg = ['MSG_INIT_SERVER_TO_CLIENT', model_name, dataset, minibatch, step_size,
                   batch_size, total_data, indices_this_node,
                   use_min_loss, node_i]
            send_msg(client_sock_all[node_i], msg)
            #print(time.time(), " node ", node_i, "MSG_INIT_SERVER_TO_CLIENT msg has send")
        print('-----------------------------------------All clients connected，init msg  has been sended-------------------------------------------------')
        # 接收消息，知道各个节点都已经收到了相对应的数据
        for node_i in range(0, n_nodes):
            recv_msg(client_sock_all[node_i], 'MSG_DATA_PREP_FINISHED_CLIENT_TO_SERVER')
            # 开始进行边缘节点的协作训练
            print("node_i :", node_i,'------------------------------------------------Ready to training----------------------------------------------------------')
        time_total_all_start = time.time()
        #正式等待client聚合
        while True:

            iter_times = iter_times + 1
            # server 数据全局模型数据传送给client
            for node_i in range(0, n_nodes):
                msg = ['MSG_WEIGHT_TAU_SERVER_TO_CLIENT', w_global, tau_config, is_last_round, prev_loss_is_min,
                       node_i, iter_times]
                send_msg(client_sock_all[node_i], msg)
                #print(time.time(), "  ", iter_times, " local iteration node ", node_i, "global msg has send")
            print("server is_last_round", is_last_round)
            if is_last_round:
                break
            # 记录t-1时刻全局模型参数值
            if not last_is_nan:
                w_global_prev = w_global
            max_local_time = 0
            w_local_all = []
            data_size_local_all = []
            datalength = 0
           # print("..................server msg have send , Waiting for the ", iter_times, " local iteration at client..................................")
            for node_i in range(0, n_nodes):
                    msg = recv_msg(client_sock_all[node_i], 'MSG_WEIGHT_TIME_SIZE_CLIENT_TO_SERVER')
                    # ['MSG_WEIGHT_TIME_SIZE_CLIENT_TO_SERVER', w, time_local, tau_actual, data_size_local]
                    if msg[2] > max_local_time:
                        max_local_time = msg[2]
                    w_local_all.append(msg[1])
                    datalength += msg[4]
                    data_size_local_all.append(msg[4])
            print("..................", iter_times," local iteration at client has finished..................................")
               #聚合
            for i in range(len(w_local_all)):
                    w_local = w_local_all[i]
                    data_size_local = data_size_local_all[i]
                    rate = float(data_size_local) / float(datalength)
                    w_global = w_global + w_local * rate
                    #print(w_local)
            if True in np.isnan(w_global):
                    w_global = w_global_prev
                    last_is_nan = True
            actual_times = actual_times + max_local_time
            print(actual_times,total_time,max_local_time)
            if actual_times >= total_time:
                is_last_round = True


        w_eval = w_global
        loss_final = model.loss(train_image, train_label, w_eval)
        accuracy_final = model.accuracy(test_image, test_label, w_eval)
        redf = pd.DataFrame(columns=["method", "n_nodes", "total_time", "minibatch", "iter_times", "tau", "loss", "accuracy"])
        redf.loc[len(redf) + 1] = ["FedAvg", n_nodes, total_time, minibatch, iter_times, tau, loss_final,
                                   accuracy_final]
        redf.to_csv(gl.PATH + 'case_3.csv', mode='a', header=False)
        print("------------------------------------------------end", str(t),"experiments-------------------------------------------------------------")









