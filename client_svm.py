import copy
import socket
import time
import struct

from control_algorithm.adaptive_tau import ControlAlgAdaptiveTauClient, ControlAlgAdaptiveTauServer
from data_reader.data_reader import get_data, get_data_train_samples
from models.get_model import get_model
from util.sampling import MinibatchSampling
from util.utils import send_msg, recv_msg
from util.tools import *
# Configurations are in a separate config.py file
from config import SERVER_ADDR, SERVER_PORT, dataset_file_path

sock = socket.socket()
sock.connect((SERVER_ADDR, SERVER_PORT))
try:
    # 接受数据
    itmes=1
    while True:
        print("client--------------------------",itmes)
        itmes+=1
        msg = recv_msg(sock, 'MSG_INIT_SERVER_TO_CLIENT')
        model_name, dataset, minmatch, step_size, batch_size, total_data, indices_this_node, use_min_loss, node_num = client_revinit_mag(
            msg)
        print("---------------------------------------------node", node_num, "read data----------------------------------------")
        model = get_model(model_name)
        train_image, train_label, _, _, _ = get_data(dataset, total_data, dataset_file_path, sim_round=0)
        sampler = MinibatchSampling(indices_this_node, batch_size, 0)
        train_indices = None
        data_size_local = len(indices_this_node)
        print("----------------------------client data length is ", data_size_local)
        # print(sampler)
        msg = ['MSG_DATA_PREP_FINISHED_CLIENT_TO_SERVER']
        send_msg(sock, msg)
        is_last_round=False
        while True:
            #从通信开始，到通信结束
            time_local_start = time.time()
            #print("client is_last_round",is_last_round)
            msg = recv_msg(sock, 'MSG_WEIGHT_TAU_SERVER_TO_CLIENT')
            # ['MSG_WEIGHT_TAU_SERVER_TO_CLIENT', w_global, tau_config, is_last_round, prev_loss_is_min]
            w = msg[1]
            tau_config = msg[2]
            is_last_round = msg[3]
            prev_loss_is_min = msg[4]
            node_i=msg[5]
            iter_times=msg[6]
            if is_last_round:
                break
            #print("iter_times ",iter_times,"node", node_i, "global msg has recieved")
            #print(len(w),tau_config,is_last_round,prev_loss_is_min)
            w_last_global = w
            tau_actual=0
            w_local=copy.deepcopy(w)
            last_is_nan=False
            w_local_prev=w_local
            while tau_actual<tau_config:
                sample_indices = sampler.get_next_batch()
                local_train_image, local_train_label = get_data_train_samples(dataset, sample_indices, dataset_file_path)
                #print("--------------------client   training data length--",len(train_label))
                train_indices = range(0, min(batch_size, len(local_train_label)))
                tau_actual += 1
                grad = model.gradient(local_train_image, local_train_label, w_local, train_indices)

                w_local = w_local - step_size * grad
                if True in np.isnan(w_local):
                    last_is_nan=True
                    print("sample_indices",sample_indices)

                #本地模型参数不为nan
                if not last_is_nan:
                    w_local_prev = w_local
            #计算准确率
            accuracy_local = model.svm_accuracy(train_image, train_label, w_local)
            time_local_end = time.time()
            time_local = time_local_end - time_local_start
            msg = ['MSG_WEIGHT_TIME_SIZE_CLIENT_TO_SERVER', w_local, time_local, tau_actual, data_size_local]
            send_msg(sock, msg)
        print("-------------------------------node", node_i, "one experiment end----------------------------")


except (struct.error, socket.error):
    print('Server has stopped')
    pass
