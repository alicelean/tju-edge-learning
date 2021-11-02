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



batch_size_prev = None
total_data_prev = None
sim_prev = None

try:
    # 接受数据
    msg = recv_msg(sock, 'MSG_INIT_SERVER_TO_CLIENT')

    # ['MSG_INIT_SERVER_TO_CLIENT', model_name, dataset, minibatch, step_size,
    #        batch_size, total_data, indices_this_node,
    #        use_min_loss]
    model_name, dataset, minmatch, step_size, batch_size, total_data, indices_this_node, use_min_loss, node_num = client_revinit_mag(
        msg)
    print("node", node_num, "read data")
    model = get_model(model_name)
    train_image, train_label, _, _, _ = get_data(dataset, total_data, dataset_file_path, sim_round=0)
    sampler = MinibatchSampling(indices_this_node, batch_size, 0)
    train_indices = None
    data_size_local = len(indices_this_node)
    print("client data length is ", data_size_local)
    # print(sampler)
    msg = ['MSG_DATA_PREP_FINISHED_CLIENT_TO_SERVER']
    send_msg(sock, msg)
    is_last_round=False
    while True:
        print("client is_last_round",is_last_round)
        if is_last_round:
            break
        msg = recv_msg(sock, 'MSG_WEIGHT_TAU_SERVER_TO_CLIENT')
        # ['MSG_WEIGHT_TAU_SERVER_TO_CLIENT', w_global, tau_config, is_last_round, prev_loss_is_min]
        w = msg[1]
        tau_config = msg[2]
        is_last_round = msg[3]
        prev_loss_is_min = msg[4]
        node_i=msg[5]
        iter_times=msg[6]
        print("iter_times ",iter_times,"node", node_i, "global msg has recieved")
        #print(len(w),tau_config,is_last_round,prev_loss_is_min)
        w_last_global = w
        tau_actual=0
        w_local=copy.deepcopy(w)
        time_local_start= time.time()
        last_is_nan=False
        w_local_prev=w_local
        #记录grad
        grad_list=[]
        while tau_actual<tau_config:
            sample_indices = sampler.get_next_batch()
            local_train_image, local_train_label = get_data_train_samples(dataset, sample_indices, dataset_file_path)
            #print("--------------------client   training data length--",len(train_label))
            train_indices = range(0, min(batch_size, len(local_train_label)))
            tau_actual += 1
            grad = model.gradient(local_train_image, local_train_label, w_local, train_indices)
            grad_list.append(grad)
            w_local = w_local - step_size * grad
            if True in np.isnan(w_local):
                last_is_nan=True
                print("sample_indices",sample_indices)

            #本地模型参数不为nan
            if not last_is_nan:
                w_local_prev = w_local

        time_local_end = time.time()
        time_local = time_local_end - time_local_start
        #计算准确率
        accuracy_local = model.accuracy(train_image, train_label, w_local)
        msg = ['MSG_WEIGHT_TIME_SIZE_CLIENT_TO_SERVER', w_local, time_local, tau_actual, data_size_local,grad_list[0]]
        send_msg(sock, msg)


#         batch_size_prev = batch_size
#         total_data_prev = total_data


#         last_batch_read_count = None
#
#
#
#         if isinstance(control_alg_server_instance, ControlAlgAdaptiveTauServer):
#             control_alg = ControlAlgAdaptiveTauClient()
#         else:
#             control_alg = None
#
#         w_prev_min_loss = None
#         w_last_global = None
#         total_iterations = 0
#

#
#         while True:
#             print('---------------------------------------------------------------------------')
#
#             msg = recv_msg(sock, 'MSG_WEIGHT_TAU_SERVER_TO_CLIENT')
#             # ['MSG_WEIGHT_TAU_SERVER_TO_CLIENT', w_global, tau, is_last_round, prev_loss_is_min]
#             w = msg[1]
#             tau_config = msg[2]
#             is_last_round = msg[3]
#             prev_loss_is_min = msg[4]
#
#             if prev_loss_is_min or ((w_prev_min_loss is None) and (w_last_global is not None)):
#                 w_prev_min_loss = w_last_global
#
#             if control_alg is not None:
#                 control_alg.init_new_round(w)
#
#             time_local_start = time.time()  #Only count this part as time for local iteration because the remaining part does not increase with tau
#
#             # Perform local iteration
#             grad = None
#             loss_last_global = None   # Only the loss at starting time is from global model parameter
#             loss_w_prev_min_loss = None
#
#             tau_actual = 0
#
#             for i in range(0, tau_config):
#
#                 # When batch size is smaller than total data, read the data here; else read data during client init above
#                 if batch_size < total_data:
#                     # When using the control algorithm, we want to make sure that the batch in the last local iteration
#                     # in the previous round and the first iteration in the current round is the same,
#                     # because the local and global parameters are used to
#                     # estimate parameters used for the adaptive tau control algorithm.
#                     # Therefore, we only change the data in minibatch when (i != 0) or (sample_indices is None).
#                     # The last condition with tau <= 1 is to make sure that the batch will change when tau = 1,
#                     # this may add noise in the parameter estimation for the control algorithm,
#                     # and the amount of noise would be related to NUM_ITERATIONS_WITH_SAME_MINIBATCH.
#
#                     if (not isinstance(control_alg, ControlAlgAdaptiveTauClient)) or (i != 0) or (train_indices is None) \
#                             or (tau_config <= 1 and
#                                 (last_batch_read_count is None or
#                                  last_batch_read_count >= num_iterations_with_same_minibatch_for_tau_equals_one)):
#
#                         sample_indices = sampler.get_next_batch()
#
#                         if read_all_data_for_stochastic:
#                             train_indices = sample_indices
#                         else:
#                             train_image, train_label = get_data_train_samples(dataset, sample_indices, dataset_file_path)
#                             train_indices = range(0, min(batch_size, len(train_label)))
#
#                         last_batch_read_count = 0
#
#                     last_batch_read_count += 1
#
#                 grad = model.gradient(train_image, train_label, w, train_indices)
#
#                 if i == 0:
#                     try:
#                         # Note: This has to follow the gradient computation line above
#                         loss_last_global = model.loss_from_prev_gradient_computation()
#                         print('*** Loss computed from previous gradient computation')
#                     except:
#                         # Will get an exception if the model does not support computing loss
#                         # from previous gradient computation
#                         loss_last_global = model.loss(train_image, train_label, w, train_indices)
#                         print('*** Loss computed from data')
#
#                     w_last_global = w
#
#                     if use_min_loss:
#                         if (batch_size < total_data) and (w_prev_min_loss is not None):
#                             # Compute loss on w_prev_min_loss so that the batch remains the same
#                             loss_w_prev_min_loss = model2.loss(train_image, train_label, w_prev_min_loss, train_indices)
#
#                 w = w - step_size * grad
#
#                 tau_actual += 1
#                 total_iterations += 1
#
#                 if control_alg is not None:
#                     is_last_local = control_alg.update_after_each_local(i, w, grad, total_iterations)
#
#                     if is_last_local:
#                         break
#
#             # Local operation finished, global aggregation starts
#             time_local_end = time.time()
#             time_all_local = time_local_end - time_local_start
#             print('time_all_local =', time_all_local)
#
#             if control_alg is not None:
#                 control_alg.update_after_all_local(model, train_image, train_label, train_indices,
#                                                    w, w_last_global, loss_last_global)
#
#             msg = ['MSG_WEIGHT_TIME_SIZE_CLIENT_TO_SERVER', w, time_all_local, tau_actual, data_size_local,
#                    loss_last_global, loss_w_prev_min_loss]
#             send_msg(sock, msg)
#
#             if control_alg is not None:
#                 control_alg.send_to_server(sock)
#
#             if is_last_round:
#                 break
#
except (struct.error, socket.error):
    print('Server has stopped')
    pass
