import copy
import time
import struct
from data_reader.data_reader import get_data, get_data_train_samples
from util.sampling import MinibatchSampling
from data_reader.data_reader import *
from util.utils import *
from util.tools import *
from config import *
# model 是svm
dataset = 'MNIST_ORIG_EVEN_ODD'  # Use for SVM model
model_name = 'ModelSVMSmooth'
control_param_phi = 0.025

#model 是cnn
# dataset = 'MNIST_ORIG_ALL_LABELS'  # Use for CNN model
# model_name = 'ModelCNNMnist'
# control_param_phi = 0.00005   # Good for CNN
train_image, train_label, test_image, test_label, train_label_orig = get_minist_data(dataset, total_data,dataset_file_path)
all_time=5000
try:
    model=createmodel(model_name,step_size)
    dim_w = model.get_weight_dimension(train_image, train_label)
    w= model.get_init_weight(dim_w, rand_seed=0)
    train_image, train_label, _, _, _ = get_data(dataset, 60000, dataset_file_path, sim_round=0)
    sampler = MinibatchSampling(train_label_orig, batch_size, 0)
    train_indices = None
    data_size_local = len(train_label_orig)
    print("client data length is ", data_size_local)
    is_last_round=False
    iter_times=0
    time_local=0
    time_local_start = time.time()
    while time_local<all_time:
        #print("client is_last_round",is_last_round)
        if is_last_round:
            break
        iter_times+=1
        w_prev = w
        print(" traning iter_times ",iter_times)
        #print(len(w),tau_config,is_last_round,prev_loss_is_min)
        last_is_nan=False
        sample_indices = sampler.get_next_batch()
        local_train_image, local_train_label = get_data_train_samples(dataset, sample_indices, dataset_file_path)
        train_indices = range(0, min(batch_size, len(local_train_label)))
        grad = model.gradient(local_train_image, local_train_label, w, train_indices)
        w= w - step_size * grad
        if True in np.isnan(w):
            last_is_nan=True
            print("sample_indices",sample_indices)
            #本地模型参数不为nan
        if not last_is_nan:
            w_prev = w

        time_local_end = time.time()
        time_local = time_local_end - time_local_start
        #计算准确率
    accuracy_local = model.accuracy(train_image, train_label, w)
    print("ceteralized accuracy_local",accuracy_local)

except (struct.error, socket.error):
    print('Server has stopped')
    pass
