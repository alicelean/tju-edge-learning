import time
import numpy as np

from data_reader.data_reader import get_data
from models.get_model import get_model
from statistic.collect_stat import CollectStatistics
from util.sampling import MinibatchSampling
from config import *
import pandas as pd
import result_value.value as gl
# model 是svm
dataset = 'MNIST_ORIG_EVEN_ODD'  # Use for SVM model
model_name = 'ModelSVMSmooth'
control_param_phi = 0.025
minibatch=3
aggre_type="center"
model = get_model(model_name)
if hasattr(model, 'create_graph'):
    model.create_graph(learning_rate=step_size)
total_time = 1000    # Actual total time, where use_fixed_averaging_slots has no effect
time_list=[]

for total_time in time_list:
    train_image, train_label, test_image, test_label, train_label_orig = get_data(dataset, total_data,dataset_file_path)
    #初始化数据
    sampler = MinibatchSampling(np.array(range(0, len(train_label))), batch_size, 0)
    dim_w = model.get_weight_dimension(train_image, train_label)
    w_init = model.get_init_weight(dim_w, rand_seed=0)
    all_cost_time = 0  # Recomputed total time using estimated time for each local and global update,                           # using predefined values when use_fixed_averaging_slots = true
    w = w_init
    w_min_loss = None
     # Loop for multiple rounds of local iterations + global aggregation
    itimers=0
    loss_min = np.inf
    print('-------------------------Start learning-----------------------')
    while True:
        itimers += 1
        time_start = time.time()
        w_prev = w
        train_indices = sampler.get_next_batch()
        grad = model.gradient(train_image, train_label, w, train_indices)
        w = w - step_size * grad
        if True in np.isnan(w):
            print('*** w_global is NaN, using previous value')
            w = w_prev   # If current w_global contains NaN value, use previous w_global
        # Calculate time
        time_end = time.time()
        print(itimers, "grad is", len(grad), "train_indices is", len(train_indices))
        cost_time = time_end - time_start
        each_time = max(0.0, cost_time)
        print('Time for one local iteration:', each_time)
        # Compute time in current slot
        all_cost_time += each_time
        print('Time for one local iteration:', each_time, "cost time ", all_cost_time)
        # Check remaining resource budget, stop if exceeding resource budget
        if all_cost_time >= total_time:
            break
    print('-------------------------End learning-----------------------')
    w_eval=w
    accuracy_final = model.accuracy(train_image, train_label, w_eval)
    accuracy_local2 = model.accuracy(train_image, test_label, w_eval)
    loss_final = model.loss(train_image, train_label, w_eval)
    print("ceteralized ", model_name, "accuracy_local", accuracy_final, "accuracy_local2 ", accuracy_local2)
    n_nodes=1
    redf = pd.DataFrame(columns=["method", "n_nodes", "total_time", "minibatch", "iter_times", "tau", "loss", "accuracy"])
    tau=itimers
    redf.loc[len(redf) + 1] = [aggre_type, n_nodes, total_time, minibatch, itimers, tau, loss_final, accuracy_final]
    # for tau in range(0,1000,5):
    #     redf.loc[len(redf) + 1] = [aggre_type, n_nodes, total_time, minibatch, itimers, tau, loss_final,accuracy_final]
    redf.to_csv(gl.PATH + "center/"+ model_name + '.csv', mode='a',header=False)
