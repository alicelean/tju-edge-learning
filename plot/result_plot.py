from plot.methods import *
from data_reader.data_reader import *
from util.utils import *
import numpy as np
from util.tools import *
from config import *
import result_value.value as gl
import  pandas as pd
from matplotlib import pyplot as plt
case_list=["case1","case2","case3","case4","case5"]
aggre_type_list=["am","avg"]
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

#1.数据集绘制(获得各个节点的样本标签分布情况)-------------------------
#数据分布绘图case1
# plot_dataset_case("case1")
#数据分布绘图case2
# plot_dataset_case("case2")
#数据分布绘图case3
# plot_dataset_case("case3")
#数据分布绘图case4
plot_dataset_case("case4")
#数据分布绘图case5
# plot_dataset_case("case5")
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



#2.统计不同的资源下，tau的变化情况，先只考虑一种资源情况下total_time=10------------------------------------------------------------------------------------------
# case_type="case3"
# aggre_type="avg"
# plot_tau_accruacy(case_type,model_name,aggre_type)
#------------------------------------------------------------------------------------------------------------------------------------------------------



# 3.统计不同的资源下，tau的变化情况，先只考虑一种资源情况下total_time=10------------------------------------------------------------------------------------------
# aggre_type="disag"
# for case_type in case_list:
#     plot_tau_accruacy(case_type,model_name,aggre_type)
# aggre_type="avg"
# case_list=['case5']
# mo_name='svm/'
# # model_name = 'ModelCNNMnist'
# for case_type in case_list:
#     plot_tau_accruacy(case_type,model_name,aggre_type,mo_name)
#------------------------------------------------------------------------------------------------------------------------------------------------------




#4.数据集各个节点与总体差异的距离计算(不同距离计算公式会获得不同的文件在distance文件夹中，case_all_distance.csv)-----------------------------------------
# 欧式（Euc），标准欧式（Sde），麦哈顿（Manh），切比雪夫（Chb），闵可夫斯基（Mik），马氏（Mab）
# 'Euc','Chb','Mik','Manh','Sde','Mab'
#distance_list = ['Euc','Chb','Mik','Kl']
# distance_list = ['Js']
# case_list=['case5']
# for dis_type in distance_list:
#     for case_type in case_list:
#         get_data_difference(case_type,dis_type)
#------------------------------------------------------------------------------------------------------------------------------------------------------



#5.绘制不同距离计算下的节点总体与数据集差异----------------------------------------------------------------------------------------------
# distance_list = ['Euc','Chb','Mik']
# color_list = ["blue", "green", "red"]
# distance_list = ['Js','Kl']
# color_list = ["blue", "green"]
# #isInner 标记是否是各个节点之间的差异还是与总体的差异
# isInner=False
# for i in range(len(distance_list)):
#     dis_type=distance_list[i]
#     color=color_list[i]
#     print(dis_type,color)
#     plot_case_distance(dis_type,color,0.5,True,False,isInner)
#------------------------------------------------------------------------------------------------------------------------------------------------------




#6.计算某个数据集（nodei 与nodej）各个节点分布的数据集内在差异---------------------------------------------------------------------------------------------------------------
#生成文件：casei.csv,距离文件
# distance_list = ['Euc','Chb','Mik','Js','Kl']
#(1)生成(dis_type)_each_dis.csv,(dis_type)_inner_distance.csv 存放在inner_dis文件夹中
# ,case,node,node1_dis,node2_dis,node3_dis,node4_dis,node5_dis
# ,distype,casetype,distance
distance_list = ['Js','Kl']
for dis_type in distance_list:
    for case_type in case_list:
        get_case_difference(case_type,dis_type)


# #(2)计算节点内部差异导致的权重
# for dis_type in distance_list:
#     for case_type in case_list:
#         get_inner_dis_wegiht(case_type,dis_type)

#------------------------------------------------------------------------------------------------------------------------------------------------------




#7.绘制不同距离计算下的各个节点分布的数据集内在差异---------------------------------------------------------------------------------------------
# distance_list = ['Euc','Chb','Mik']
# color_list = ["blue", "green", "red"]
# isInner=False
# isNormal=True
# for i in range(len(distance_list)):
#     dis_type=distance_list[i]
#     color=color_list[i]
#     print(dis_type,color)
#     plot_case_distance(dis_type,color,0.5,isNormal,False,isInner)
#---------------------------------------------------------------------------#---------------------------------------------------------------------------


#8.不同距离下的各节点的权重#---------------------------------------------------------------------------
# distance_list = ['Euc','Chb','Mik']
# for case_type in case_list:
#     for dis_type in distance_list:
#         get_wegiht(dis_type,case_type)

#---------------------------------------------------------------------------


#9.绘制 不同模型、不同tau, 联邦学习accuracy与集中式的accuracy的对比图----------------#---------------------------------------------------------------------------
# model_name = 'ModelCNNMnist'
# aggre_type_list=['avg']
# case_list=['case32']
# mo_name='svm/'
# plot_compare(case_list,model_name,aggre_type_list,mo_name)
#---------------------------------------------------------------------------#---------------------------------------------------------------------------



#10.不同case下，不同聚合方式下，tau与accuracy之间的变化关系#---------------------------------------------------------------------------
# 'am','disag','indisag','feddis','jsag','klag','avg','fed_dis','jsag'
#,"red","purple",, "black"

# aggre_type_list=['avg','fedjs','js']
# case_list=['case4']
# mo_name='svm/'
# colist = ["indigo","red","blue", "green"]
# time_list=get_time_list(case_list,aggre_type_list,mo_name)
# print("time resource is:",time_list)
# plot_multicompare_of_case(time_list,case_list,aggre_type_list,colist,mo_name)

# "blue",
# aggre_type_list=['jsag','avg','disag']
# colist = ["indigo","red", "green","black"]
# case_list=["case2","case3"]
# time_list=get_time_list(case_list,aggre_type_list)
# print("time resource is:",time_list)
#plot_multicompare_of_case(time_list,case_list,aggre_type_list,colist)

#---------------------------------------------------------------------------

















