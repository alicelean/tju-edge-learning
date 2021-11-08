import time
import numpy as np

from data_reader.data_reader import get_data
from models.get_model import get_model
from statistic.collect_stat import CollectStatistics
from util.sampling import MinibatchSampling
from config import *
# model 是svm
dataset = 'MNIST_ORIG_EVEN_ODD'  # Use for SVM model
model_name = 'ModelSVMSmooth'
control_param_phi = 0.025

#model 是cnn
# dataset = 'MNIST_ORIG_ALL_LABELS'  # Use for CNN model
# model_name = 'ModelCNNMnist'
# control_param_phi = 0.00005   # Good for CNN

model = get_model(model_name)
if hasattr(model, 'create_graph'):
    model.create_graph(learning_rate=step_size)

if time_gen is not None:
    use_fixed_averaging_slots = True
else:
    use_fixed_averaging_slots = False




train_image, train_label, test_image, test_label, train_label_orig = get_data(dataset, total_data,
                                                                                      dataset_file_path)
sampler = MinibatchSampling(np.array(range(0, len(train_label))), batch_size, 0)
dim_w = model.get_weight_dimension(train_image, train_label)
w_init = model.get_init_weight(dim_w, rand_seed=0)
w = w_init
w_min_loss = None
loss_min = np.inf
print('Start learning')
total_time = 0      # Actual total time, where use_fixed_averaging_slots has no effect
total_time_recomputed = 0  # Recomputed total time using estimated time for each local and global update,                           # using predefined values when use_fixed_averaging_slots = true
it_each_local = None
 # Loop for multiple rounds of local iterations + global aggregation
itimers=1
while True:
    time_total_all_start = time.time()
    w_prev = w
    train_indices = sampler.get_next_batch()
    grad = model.gradient(train_image, train_label, w, train_indices)
    print(itimers,"grad is",len(grad),"train_indices is",len(train_indices))
    itimers+=1
    w = w - step_size * grad

    if True in np.isnan(w):
        print('*** w_global is NaN, using previous value')
        w = w_prev   # If current w_global contains NaN value, use previous w_global

        # Calculate time
    time_total_all_end = time.time()
    time_total_all = time_total_all_end - time_total_all_start
    time_one_iteration_all = max(0.0, time_total_all)

    print('Time for one local iteration:', time_one_iteration_all)

    if use_fixed_averaging_slots:
        it_each_local = max(0.00000001, time_gen.get_local(1)[0])
    else:
        it_each_local = max(0.00000001, time_one_iteration_all)

        # Compute number of iterations is current slot
    total_time_recomputed += it_each_local

        # Compute time in current slot
    total_time += time_total_all

         # Check remaining resource budget, stop if exceeding resource budget
    if total_time_recomputed >= max_time:
        break


accuracy_local = model.accuracy(train_image, train_label, w)
print("ceteralized ",model_name ,"accuracy_local",accuracy_local)
