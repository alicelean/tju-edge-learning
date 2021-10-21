# -*- coding:utf-8 -*-
import numpy as np
import pickle, struct, socket, math

#共七个函数
#处理label
#get_even_odd_from_one_hot_label：将一个10维向量label，变成奇偶标记（1，-1）
#get_index_from_one_hot_label：获取一个10维向量label对应类别的下表index值
#get_one_hot_from_label_index：根据label的index值重新获得一个10维向量的label向量
#接送数据
#send_msg：
#recv_msg：

# adptive算法使用
#moving_average：

#划分数据集
#get_indices_each_node_case：


def get_even_odd_from_one_hot_label(label):
        '''
        10维向量，变成奇偶标记（1，-1）=（偶数，奇数）
        :param label:
        :return: 奇偶标记（1，-1）=（偶数，奇数）
        '''
        for i in range(0, len(label)):
            if label[i] == 1:
                c = i % 2
                if c == 0:
                    c = 1
                elif c == 1:
                    c = -1
                return c

def get_index_from_one_hot_label(label):
        '''
        获取对应类别的下表index值
        :param label:[0,1,0,0,0,0,0,0,0,0]
        :return:[1]
        '''
        for i in range(0, len(label)):
            if label[i] == 1:
                return [i]

def get_one_hot_from_label_index(label, number_of_labels=10):
        '''
        根据label的index值重新获得一个10维向量的label向量
        :param label: （0-9）
        :param number_of_labels: 类别个数
        :return: number_of_labels维向量，默认是10维向量
        '''
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
    print(msg[0], 'sent to', sock.getpeername())


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
    print(msg[0], 'received from', sock.getpeername())

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


def get_indices_each_node_case(n_nodes,maxCase,label_list):

    '''
    :param n_nodes: 节点个数
    :param maxCase:数据集种类，只能为4
    :param label_list: 标签list
    :return: indices_each_node_case[maxCase,n_nodes]
    '''
    indices_each_node_case = []
    try:
        for i in range(0, maxCase):
            indices_each_node_case.append([])

        for i in range(0, n_nodes):
            for j in range(0, maxCase):
                indices_each_node_case[j].append([])

        # indices_each_node_case is a big list that contains N-number of sublists.
        # Sublist n contains the indices that should be assigned to node n

        min_label = min(label_list)
        max_label = max(label_list)
        #类别个数num_labels
        num_labels = max_label - min_label + 1

        print("dataset total label_list len is :" ,len(label_list))

        for i in range(0, len(label_list)):

            # case 1,随机放置到节点
            indices_each_node_case[0][(i % n_nodes)].append(i)


            # case 2
            tmp_target_node = int((label_list[i] - min_label) % n_nodes)

            #节点个数大于类别个数
            if n_nodes > num_labels:
                tmp_min_index = 0
                tmp_min_val = math.inf
                #处理每一个节点
                for n in range(0, n_nodes):
                    if n % num_labels == tmp_target_node and len(indices_each_node_case[1][n]) < tmp_min_val:
                        tmp_min_val = len(indices_each_node_case[1][n])
                        tmp_min_index = n
                tmp_target_node = tmp_min_index
            indices_each_node_case[1][tmp_target_node].append(i)



            # case 3，每个节点都是所有的数据集。
            for n in range(0, n_nodes):
                indices_each_node_case[2][n].append(i)



            # case 4
            #np.ceil函数返回数字的上入整数,tmp是一个固定的数值
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
    except Exception as e:
        print("util.utils.get_indices_each_node_case Error :",e)

    return indices_each_node_case
