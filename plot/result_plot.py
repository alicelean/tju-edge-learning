from plot.methods import *
from data_reader.data_reader import *
from util.utils import *
import numpy as np
from util.tools import *
from config import *
import result_value.value as gl
import  pandas as pd
from matplotlib import pyplot as plt


# plt.plot(x_list, y_list, linewidth=3, color='red', marker='o', linestyle='--', label='我是图例')
# plt.legend(loc='upper left')  # loc设置图例位置
# # 设置图标的标题，并且给坐标轴加上标签
# plt.title('我是标题', fontsize=20)  # fontsize 修改标题大小
# plt.xlabel('我是横轴')
# plt.ylabel('我是纵轴')
# plt.show()






#不同资源的绘制
#DF=pd.read_csv(gl.PATH+'case_1.csv')
# print(DF.head())
# colname=[column for column in DF]
# #列名
# time_list=list(set(DF['total_time'].tolist()))
#different_time(time_list,DF)
#

# DF2=pd.read_csv(gl.PATH+'case_2.csv')
# colname=[column for column in DF2]
# print(colname)
# time_list=list(set(DF2['total_time'].tolist()))
# different_time(time_list,DF2)


# DF2=pd.read_csv(gl.PATH+'case_4.csv')
# colname=[column for column in DF2]
# print(colname)
# time_list=list(set(DF2['total_time'].tolist()))
# different_time(time_list,DF2)

#数据集绘制-------------------------
#数据分布绘图case1
# plot_dataset_case("case1")
#数据分布绘图case2
# plot_dataset_case("case2")
#数据分布绘图case3
# plot_dataset_case("case3")

#-----------------------------------
# case_type="case2"
# casepath=gl.PATH +case_type+"_"+model_name+'_tau.csv'
# DF2=pd.read_csv(casepath)
# colname=[column for column in DF2]
# print(colname)
# #不同资源
# time_list=list(set(DF2['total_time'].tolist()))
# print(time_list)
# different_time(time_list,DF2)

case_type="case2"
casepath=gl.PATH +case_type+"_"+model_name+'_tau.csv'
DF2=pd.read_csv(casepath)
colname=[column for column in DF2]
print(colname)
#不同资源
time_list=list(set(DF2['total_time'].tolist()))
print(time_list)
different_time(case_type,model_name,time_list,DF2)