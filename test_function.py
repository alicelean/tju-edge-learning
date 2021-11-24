from plot.methods import *
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
# case1 = [0.5]
# case2=[1.5]
# case3=[2.5]
# Y = [20]
# YY=[23]
# YYY=[30]
# #fig = plt.figure()
# plt.bar(case1, Y, 1, color="blue")
# plt.bar(case2,YY,1,color="green") #使用不同颜色
# plt.bar(case3,YYY,1,color="red") #使用不同颜色
# plt.xlabel("X-axis") #设置X轴Y轴名称
# plt.ylabel("Y-axis")
# plt.title("bar chart")
# #使用text显示数值
# for a,b in zip(case1,Y):
#     print("aa",a,b)
#     plt.text(a, b+0.05, '%.0f' % b, ha='center', va= 'bottom',fontsize=11)
# for a,b in zip(case2,YY):
#     plt.text(a, b+0.05, '%.0f' % b, ha='center', va= 'bottom',fontsize=11)
# for a,b in zip(case3,YYY):
#     plt.text(a, b+0.05, '%.0f' % b, ha='center', va= 'bottom',fontsize=11)
# plt.ylim(0,40) #设置Y轴上下限
# plt.show()




Y_list=[0.63,2.3,1]
Xaxis="test"
Yaxis="hig"
title="pic"
color_list=["blue","green","red"]
plot_bar_color_value(Y_list,color_list,Xaxis,Yaxis,title,ymax=3)