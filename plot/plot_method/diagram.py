#曲线图
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm  #字体管理器
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def line():
    '''
    单条折线
    :return:
    '''
    x_data = ['2011', '2012', '2013', '2014', '2015', '2016', '2017']
    y_data = [58000, 60200, 63000, 71000, 84000, 90500, 107000]
    plt.plot(x_data, y_data)
    plt.show()


def multiline():
    '''
    多条折线，x轴下表一致
    对于复式折线图，应该为每条折线添加图例，可以通过legend()函数来实现
    color  ------  指定折线的颜色
    linewidth   --------  指定折线的宽度
    linestyle   --------  指定折线的样式
    ‘  - ’ ： 表示实线
    ’ - - ‘   ：表示虚线
    ’ ：  ‘：表示点线
    ’ - . ‘  ：表示短线、点相间的虚线
    :return:
    '''
    x_data = ['2011', '2012', '2013', '2014', '2015', '2016', '2017']
    y_data = [58000, 60200, 63000, 71000, 84000, 90500, 107000]
    y_data2 = [52000, 54200, 51500, 58300, 56800, 59500, 62700]
    plt.title("电子产品销售量")
    #设置字体
    #my_font = fm.FontProperties(fname="/usr/share/fonts/wqy-microhei/wqy-microhei.ttc")
    #设置每一条曲线的样式，颜色，形状，宽度，图例信息
    ln1, = plt.plot(x_data, y_data, color='red', linewidth=2.0, linestyle='--')
    ln2, = plt.plot(x_data, y_data2, color='blue', linewidth=3.0, linestyle='-.')
    #plt.legend(handles=[ln1, ln2], labels=['鼠标的年销量', '键盘的年销量'], prop=my_font)
    #大图的相关信息
    #plt.title("电子产品销售量", fontproperties=my_font)  # 设置标题及字体
    plt.legend(handles=[ln1, ln2], labels=['鼠标的年销量', '键盘的年销量'])
    ax = plt.gca()
    ax.spines['right'].set_color('none')  # right边框属性设置为none 不显示
    ax.spines['top'].set_color('none')  # top边框属性设置为none 不显示
    plt.show()


def multilinePicture():
    '''
    多个曲线子图
    :return:
    '''
    #画布
    plt.figure()
    #设置字体
    my_font = fm.FontProperties(fname="/usr/share/fonts/wqy-microhei/wqy-microhei.ttc")
    x_data = np.linspace(-np.pi, np.pi, 64, endpoint=True)
    gs = gridspec.GridSpec(2, 3)  # 将绘图区分成两行三列，方便设置子图占领的位置大小
    ax1 = plt.subplot(gs[0, :])  # 指定ax1占用第一行(0)整行
    ax2 = plt.subplot(gs[1, 0])  # 指定ax2占用第二行(1)的第一格(第二个参数为0)
    ax3 = plt.subplot(gs[1, 1:3])  # 指定ax3占用第二行(1)的第二、三格(第二个参数为1：3)
    # 绘制正弦曲线
    ax1.plot(x_data, np.sin(x_data))
    ax1.spines['right'].set_color('none')
    ax1.spines['top'].set_color('none')
    ax1.spines['bottom'].set_position(('data', 0))
    ax1.spines['left'].set_position(('data', 0))
    ax1.set_title('正弦曲线', fontproperties=my_font)

    # 绘制余弦曲线
    ax2.plot(x_data, np.cos(x_data))
    ax2.spines['right'].set_color('none')
    ax2.spines['top'].set_color('none')
    ax2.spines['bottom'].set_position(('data', 0))
    ax2.spines['left'].set_position(('data', 0))
    ax2.set_title('余弦曲线', fontproperties=my_font)

    # 绘制正切曲线
    ax3.plot(x_data, np.tan(x_data))
    ax3.spines['right'].set_color('none')
    ax3.spines['top'].set_color('none')
    ax3.spines['bottom'].set_position(('data', 0))
    ax3.spines['left'].set_position(('data', 0))
    ax3.set_title('正切曲线', fontproperties=my_font)
    plt.show()


def plot_two_line():
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    #画布
    plt.figure()
    x = pd.Series(np.exp(np.arange(20)))
    x2 = pd.Series(np.log10(x))  # np.log()是以e为底的
    p1 = x.plot(label=u'原始数据图')
    p2 = x2.plot(secondary_y=True, style='--', color='r', )
    # print(x)
    plt.ylabel('normal')
    #用于设置和获取y轴当前刻度位置和标签
    plt.yticks(np.arange(0, 60, 5))
    plt.xticks(np.arange(0, 5, 1))
    plt.ylabel('指数坐标')
    blue_line = mlines.Line2D([], [], linestyle='-', color='blue', markersize=2, label=u'原始数据图')
    red_line = mlines.Line2D([], [], linestyle='--', color='red', markersize=2, label=u'对数数据图')
    plt.legend(handles=[blue_line, red_line], loc='upper left')
    plt.grid(True)
    plt.show()

plot_two_line()



