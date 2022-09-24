import time
import numpy as np

from data_reader.data_reader import get_data
from models.get_model import get_model
from statistic.collect_stat import CollectStatistics
from util.sampling import MinibatchSampling
from config import *
from util.aggregator import *
from config import *
from plot.methods import *
from util.tools import *
from util.utils import *

# model 是cnn
dataset = 'MNIST_ORIG_ALL_LABELS'  # Use for CNN model
model_name = 'ModelCNNMnist'
control_param_phi = 0.00005  # Good for CNN
model = createmodel(model_name, step_size)
minibatch = 3
total_time = 10
# 读取数据
train_image, train_label, test_image, test_label, train_label_orig = get_minist_data(dataset, total_data,
                                                                                     dataset_file_path)
sampler = MinibatchSampling(np.array(range(0, len(train_label))), batch_size, 0)
dim_w = model.get_weight_dimension(train_image, train_label)
w_init = model.get_init_weight(dim_w, rand_seed=0)
w = w_init
w_min_loss = None
loss_min = np.inf
print('Start learning')
all_cost_time = 0  # Actual total time, where use_fixed_averaging_slots has no effect
it_each_local = None
# Loop for multiple rounds of local iterations + global aggregation
itimers = 0
while True:
    itimers += 1
    time_start = time.time()
    # 记录前一次的权重参数
    w_prev = w
    # 取batch数据
    train_indices = sampler.get_next_batch()
    # 计算梯度，并更新
    grad = model.gradient(train_image, train_label, w, train_indices)
    w = w - step_size * grad
      # 对异常参数进行处理
    if True in np.isnan(w):
        print('*** w_global is NaN, using previous value')
        w = w_prev  # If current w_global contains NaN value, use previous w_global

        # Calculate time
    # 判定是否要进行终止
    time_end = time.time()
    print(itimers, "grad is", len(grad), "train_indices is", len(train_indices))
    cost_time = time_end - time_start
    each_time = max(0.0, cost_time)

    # Compute number of iterations is current slot
    all_cost_time += each_time
    print('Time for one local iteration:', each_time, "cost time ", all_cost_time)
    # Check remaining resource budget, stop if exceeding resource budget
    if all_cost_time >= total_time:
        break
    # if itimers>3855:
    #     break

w_eval=w
accuracy_final = model.accuracy(train_image, train_label, w_eval)
accuracy_local2 = model.accuracy(test_image, test_label, w_eval)
loss_final = model.loss(train_image, train_label, w_eval)
print("ceteralized ", model_name, "accuracy_local", accuracy_final, "accuracy_local2 ", accuracy_local2)

aggre_type="center"
n_nodes=1

redf = pd.DataFrame(columns=["method", "n_nodes", "total_time", "minibatch", "iter_times", "tau", "loss", "accuracy"])
tau=itimers
redf.loc[len(redf) + 1] = [aggre_type, n_nodes, total_time, minibatch, itimers, tau, loss_final, accuracy_final]
# for tau in range(0,1000,5):
#     redf.loc[len(redf) + 1] = [aggre_type, n_nodes, total_time, minibatch, itimers, tau, loss_final,accuracy_final]
redf.to_csv(gl.PATH + "center/"+ model_name + '.csv', mode='a',header=False)