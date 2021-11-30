from plot.methods import *
from data_reader.data_reader import *
from util.utils import *
import numpy as np
from util.tools import *
from config import *
import result_value.value as gl
import  pandas as pd
from matplotlib import pyplot as plt
case_list=["case1","case2","case3"]

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



#统计不同的资源下，tau的变化情况，先只考虑一种资源情况下total_time=10---------------
# case_type="case3"
# aggre_type="avg"
# plot_tau_accruacy(case_type,model_name,aggre_type)


#统计不同的资源下，tau的变化情况，先只考虑一种资源情况下total_time=10---------------
# aggre_type="avg"
# for case_type in case_list:
#     plot_tau_accruacy(case_type,model_name,aggre_type)
aggre_type="avg"
model_name = 'ModelCNNMnist'
for case_type in case_list:
    plot_tau_accruacy(case_type,model_name,aggre_type)
#---------------------------------------------------------------------------
#数据集各个节点与总体差异的距离计算
# 欧式（Euc），标准欧式（Sde），麦哈顿（Manh），切比雪夫（Chb），闵可夫斯基（Mik），马氏（Mab）
#'Euc','Chb','Mik','Manh','Sde','Mab'
# case_list=["case1","case2","case3"]
# distance_list = ['Euc','Chb','Mik']
# for dis_type in distance_list:
#     for case_type in case_list:
#         get_data_difference(case_type,dis_type)

#绘制不同距离计算下的数据集差异------------------
# distance_list = ['Euc','Chb','Mik']
# color_list = ["blue", "green", "red"]
#isInner=False
# for i in range(len(distance_list)):
#     dis_type=distance_list[i]
#     color=color_list[i]
#     print(dis_type,color)
#     plot_case_distance(dis_type,color,0.5,True,False,isInner)


#计算数据集各个节点分布的内在差异------------------------------------
# case_list=["case1","case2","case3"]
# distance_list = ['Euc','Chb','Mik']
# for dis_type in distance_list:
#     for case_type in case_list:
#         get_case_difference(case_type,dis_type)

#绘制不同距离计算下的数据集内在差异------------------
# distance_list = ['Euc','Chb','Mik']
# color_list = ["blue", "green", "red"]
# isInner=False
# isNormal=True
# for i in range(len(distance_list)):
#     dis_type=distance_list[i]
#     color=color_list[i]
#     print(dis_type,color)
#     plot_case_distance(dis_type,color,0.5,isNormal,False,isInner)

