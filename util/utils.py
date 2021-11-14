import numpy as np
import pickle, struct, socket, math
import pandas as pd
import result_value.value as gl
def get_even_odd_from_one_hot_label(label):
    for i in range(0, len(label)):
        if label[i] == 1:
            c = i % 2
            if c == 0:
                c = 1
            elif c == 1:
                c = -1
            return c


def get_index_from_one_hot_label(label):
    for i in range(0, len(label)):
        if label[i] == 1:
            return [i]


def get_one_hot_from_label_index(label, number_of_labels=10):
    one_hot = np.zeros(number_of_labels)
    one_hot[label] = 1
    return one_hot


def send_msg(sock, msg):
    '''
    传送信息
    :param sock:
    :param msg:
    :return:
    '''
    msg_pickle = pickle.dumps(msg)
    sock.sendall(struct.pack(">I", len(msg_pickle)))
    sock.sendall(msg_pickle)
   # print(msg[0], 'sent to', sock.getpeername())


def recv_msg(sock, expect_msg_type=None):
    '''
    接受信息
    :param sock:
    :param expect_msg_type:
    :return:
    '''
    msg_len = struct.unpack(">I", sock.recv(4))[0]
    msg = sock.recv(msg_len, socket.MSG_WAITALL)
    msg = pickle.loads(msg)
    #print(msg[0], 'received from', sock.getpeername())

    if (expect_msg_type is not None) and (msg[0] != expect_msg_type):
        raise Exception("Expected " + expect_msg_type + " but received " + msg[0])
    return msg


def moving_average(param_mvavr, param_new, movingAverageHoldingParam):
    if param_mvavr is None or np.isnan(param_mvavr):
        param_mvavr = param_new
    else:
        if not np.isnan(param_new):
            param_mvavr = movingAverageHoldingParam * param_mvavr + (1 - movingAverageHoldingParam) * param_new
    return param_mvavr


def get_indices_each_node_case(n_nodes, maxCase, label_list):
    indices_each_node_case = []

    for i in range(0, maxCase):
        indices_each_node_case.append([])

    for i in range(0, n_nodes):
        for j in range(0, maxCase):
            indices_each_node_case[j].append([])

    # indices_each_node_case is a big list that contains N-number of sublists. Sublist n contains the indices that should be assigned to node n

    min_label = min(label_list)
    max_label = max(label_list)
    num_labels = max_label - min_label + 1

    for i in range(0, len(label_list)):
        # case 1
        indices_each_node_case[0][(i % n_nodes)].append(i)

        # case 2
        tmp_target_node = int((label_list[i] - min_label) % n_nodes)
        if n_nodes > num_labels:
            tmp_min_index = 0
            tmp_min_val = math.inf
            for n in range(0, n_nodes):
                if n % num_labels == tmp_target_node and len(indices_each_node_case[1][n]) < tmp_min_val:
                    tmp_min_val = len(indices_each_node_case[1][n])
                    tmp_min_index = n
            tmp_target_node = tmp_min_index
        indices_each_node_case[1][tmp_target_node].append(i)

        # case 3
        for n in range(0, n_nodes):
            indices_each_node_case[2][n].append(i)

        # case 4
        tmp = int(np.ceil(min(n_nodes, num_labels) / 2))
        if label_list[i] < (min_label + max_label) / 2:
            tmp_target_node = i % tmp
        elif n_nodes > 1:
            tmp_target_node = int(((label_list[i] - min_label) % (min(n_nodes, num_labels) - tmp)) + tmp)

        if n_nodes > num_labels:
            tmp_min_index = 0
            tmp_min_val = math.inf
            for n in range(0, n_nodes):
                if n % num_labels == tmp_target_node and len(indices_each_node_case[3][n]) < tmp_min_val:
                    tmp_min_val = len(indices_each_node_case[3][n])
                    tmp_min_index = n
            tmp_target_node = tmp_min_index

        indices_each_node_case[3][tmp_target_node].append(i)

    return indices_each_node_case


def get_case_1(n_nodes, label_list):
    '''
    随机放置各个节点数据
    :param n_nodes:
    :param label_list:
    :return:
    '''
    case = []
    df=pd.DataFrame(columns=["node_i","datalength"])
    for i in range(0, n_nodes):
        case.append([])
    for i in range(0, len(label_list)):
        node_i=i % n_nodes
        case[node_i].append(i)
    for node_i in range(0, n_nodes):
        print("node",node_i," data len is :",len(case[node_i]))
        df.loc[len(df)+1]=[node_i,len(case[node_i])]
    df.to_csv(gl.PATH+"datacas1.csv")
    return case


def get_case_2(n_nodes, label_list):
    '''
    有偏的放置一些数据
    :param n_nodes:
    :param label_list:
    :return:
    '''
    case= []
    #初始化
    for i in range(0, n_nodes):
        case.append([])

    for i in range(0, len(label_list)):
        if label_list[i]==7 or label_list[i]==3:
            case[0].append(i)
        elif label_list[i]==0 or label_list[i]==4:
            case[1].append(i)
        else:
            case[(i % n_nodes)].append(i)
    return case


def get_case_3(n_nodes, label_list):
    '''
   有偏的放置数据
    :param n_nodes:
    :param label_list:
    :return:
    '''
    case = []
    df=pd.DataFrame(columns=["node_i","datalength"])
    #初始化case
    for i in range(0, n_nodes):
        case.append([])
    num=0
    for i in range(0, len(label_list)):
        node_i=4
        if label_list[i] == 2 or label_list[i] == 6:
            node_i=0
        elif label_list[i] == 0 or label_list[i] == 1:
            node_i = 2
        elif label_list[i] == 7 or label_list[i] == 9:
            node_i = 3
        else:
            node_i = i % n_nodes
        case[node_i].append(i)

    for node_i in range(0, n_nodes):
        print("node",node_i," data len is :",len(case[node_i]))
        df.loc[len(df)+1]=[node_i,len(case[node_i])]
    df.to_csv(gl.PATH+"datacase3.csv")
    return case
