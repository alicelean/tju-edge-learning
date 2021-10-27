from plot.methods import *
from data_reader.data_reader import *
from util.utils import *
import numpy as np
from util.tools import *
from config import *
import result_value.value as gl
import  pandas as pd
from matplotlib import pyplot as plt
DF=pd.read_csv(gl.PATH+'case_1.csv')
DF2=pd.read_csv(gl.PATH+'case_2.csv')
#print(DF.head(10))

# plt.plot(x_list, y_list, linewidth=3, color='red', marker='o', linestyle='--', label='我是图例')
# plt.legend(loc='upper left')  # loc设置图例位置
# # 设置图标的标题，并且给坐标轴加上标签
# plt.title('我是标题', fontsize=20)  # fontsize 修改标题大小
# plt.xlabel('我是横轴')
# plt.ylabel('我是纵轴')
# plt.show()
def plot_dataset_case1():
    train_image, train_label, test_image, test_label, train_label_orig = get_minist_data(dataset, total_data,dataset_file_path)
    indices_each_node= get_case_1(n_nodes,train_label_orig)
    for j in range(len(indices_each_node)):
        y_list = np.zeros(10)
        for i in indices_each_node[j]:
           y_list[train_label_orig[i]]+=1
        print(y_list)
        plot_line(y_list,j)
#数据分布绘图
#plot_dataset_case1()
#结果绘图
def different_time(time_list,DF):
    for v in time_list:
        s=DF[DF['total_time']==v]
        title="total_time is "+str(v)
        plt.title(title, fontsize=20)
        #s.sort_index(axis=0,by='tau',ascending=True)
        data=s[['tau','accuracy']]
        plt.xlabel('each node local model update times')
        plt.ylabel('global model accuracy')
        x=data['tau'].tolist()
        y=data['accuracy'].tolist()
        plot_two(x,y)
#不同资源的绘制
# print(DF.head())
# colname=[column for column in DF]
# #列名
# time_list=list(set(DF['total_time'].tolist()))
#different_time(time_list,DF)
colname=[column for column in DF2]
print(colname)
time_list=list(set(DF2['total_time'].tolist()))
different_time(time_list,DF2)

