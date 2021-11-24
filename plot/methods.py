from matplotlib import pyplot as plt
from util.distance import *
import pandas as pd
from util.utils import *
import result_value.value as gl
from data_reader.data_reader import *
from config import *
def plot_line(data,j):
    '''
    :param data: 要描绘的数据
    :param j:
    :return:
    '''
    data.sort()
    plt.xlabel("picture label")
    plt.ylabel('label num')
    title='node'+str(j)+' data set'
    plt.title(title, fontsize=20)
    plt.bar(range(len(data)), data)
    plt.show()

def plot_line2(data,j,case_type=None):
    '''
    :param data: 要描绘的数据
    :param j:
    :return:
    '''
    #data.sort()
    plt.xlabel("picture label")
    plt.ylabel('label num')
    title='node'+str(j)+' data set'
    plt.title(title, fontsize=20)
    plt.bar(range(len(data)), data)
    plt.savefig('./picture/'+case_type+"_node"+str(j)+'.svg', format='svg')
    plt.show()

def plot_two(x,y):
    plt.plot(x, y, linewidth=3, color='red', marker='o', linestyle='--', label='我是图例')
    plt.show()

def plot_dataset_case1(case_type):
    train_image, train_label, test_image, test_label, train_label_orig = get_minist_data(dataset, total_data,dataset_file_path)
    indices_each_node= get_case_1(n_nodes,train_label_orig)
    for j in range(len(indices_each_node)):
        y_list = np.zeros(10)
        for i in indices_each_node[j]:
           y_list[train_label_orig[i]]+=1
        print(y_list)
        plot_line2(y_list,j,case_type)



def plot_dataset_case2(case_type):
    train_image, train_label, test_image, test_label, train_label_orig = get_minist_data(dataset, total_data,dataset_file_path)
    indices_each_node= get_case_2(n_nodes,train_label_orig)
    for j in range(len(indices_each_node)):
        y_list = np.zeros(10)
        for i in indices_each_node[j]:
           y_list[train_label_orig[i]]+=1
        print(y_list)
        plot_line2(y_list,j,case_type)


#结果绘图
def different_time(case_type,model_name,aggre_type,time_list,DF):
    for v in time_list:
        s=DF[DF['total_time']==v]
        title=case_type+": total_time is "+str(v)+" by "+model_name
        plt.title(title, fontsize=20)
        #s.sort_index(axis=0,by='tau',ascending=True)
        data=s[['tau','accuracy']]
        plt.xlabel('each node local model update times')
        plt.ylabel('global model accuracy '+aggre_type)
        x=data['tau'].tolist()
        y=data['accuracy'].tolist()
        plt.savefig('./picture/tau_result/' + case_type + "_" + model_name + '.svg', format='svg')
        plot_two(x,y)

def plot_dataset_case(case_type):
    '''
    绘制不同的数据的分布
    :param case_type: 表示以某种方式分布的数据集
    :return:
    '''
    train_image, train_label, test_image, test_label, train_label_orig = get_minist_data(dataset, total_data,dataset_file_path)
    if case_type == "case1":
        indices_each_node = get_case_1(n_nodes, train_label_orig)
    if case_type == "case2":
        indices_each_node = get_case_2(n_nodes, train_label_orig)
    if case_type == "case3":
        indices_each_node = get_case_3(n_nodes, train_label_orig)
    print("node_num is:",len(indices_each_node))
    node_num=len(indices_each_node)
    for node_i in range(node_num):
        node_data=indices_each_node[node_i]
        #统计各个节点上的不同label个数
        y_list = np.zeros(10)
        for label_index in node_data:
            label=train_label_orig[label_index]
            print(node_i,label_index,label)
            y_list[label]+=1
        print(node_i,y_list)
        #绘制node2的图片
        plot_line2(y_list,node_i,case_type)

# def get_case_vector(case_type):
#  '''
#  将不同的case的数据分布统计并写入 datalabel file中
#  :param case_type:
#  :return:
#  '''


def get_case_difference(case_type,dis_type=None):
    '''
    绘制不同的数据的分布的各个节点的内在差异
    :param case_type: 表示以某种方式分布的数据集
     :param dis_type: 不同方式进行距离计算，默认None表示欧式距离
     欧式（Euc），标准欧式（Sde），麦哈顿（Manh），切比雪夫（Chb），闵可夫斯基（Mik），马氏（Mab）
    :return:
    '''
    #读取数据
    train_image, train_label, test_image, test_label, train_label_orig = get_minist_data(dataset, total_data,dataset_file_path)
    if case_type == "case1":
        indices_each_node = get_case_1(n_nodes, train_label_orig)
    if case_type == "case2":
        indices_each_node = get_case_2(n_nodes, train_label_orig)
    if case_type == "case3":
        indices_each_node = get_case_3(n_nodes, train_label_orig)
    print("node_num is:",len(indices_each_node))
    node_num = len(indices_each_node)
    #存放每个节点的标签分布
    node_label_list=[]
    for node_i in range(node_num):
        # 统计各个节点上的不同label个数
        node_data = indices_each_node[node_i]
        y_list= np.zeros(10)
        for label_index in node_data:
            label=train_label_orig[label_index]
            #print(node_i,label_index,label)
            y_list[label]+=1
        node_label_list.append(y_list)
    #计算每个节点与其他节点的距离
    distance_list=[]
    for Y in node_label_list:
        for y_list in node_label_list:
            distance=alldistance(y_list, Y, dis_type)
            distance_list.append(distance)
    print(distance_list,sum(distance_list))
    sums = [dis_type, case_type, sum(distance_list)]
    df = pd.DataFrame(columns=["dis_type", "casetype", "sum"])
    df.loc[len(df) + 1] = sums
    df.to_csv(gl.PATH+"inner_dis/" + dis_type + '_inner_distance.csv', mode='a', header=False)


def get_data_difference(case_type,dis_type=None):
    '''
    绘制不同的数据的分布的与总体分布的具体距离差异
    :param case_type: 表示以某种方式分布的数据集
     :param dis_type: 不同方式进行距离计算，默认None表示欧式距离
     欧式（Euc），标准欧式（Sde），麦哈顿（Manh），切比雪夫（Chb），闵可夫斯基（Mik），马氏（Mab）
    :return:
    '''
    #读取数据
    train_image, train_label, test_image, test_label, train_label_orig = get_minist_data(dataset, total_data,dataset_file_path)
    if case_type == "case1":
        indices_each_node = get_case_1(n_nodes, train_label_orig)
    if case_type == "case2":
        indices_each_node = get_case_2(n_nodes, train_label_orig)
    if case_type == "case3":
        indices_each_node = get_case_3(n_nodes, train_label_orig)
    print("node_num is:",len(indices_each_node))
    node_num = len(indices_each_node)
    #计算总体分布：定义map,统计value的个数
    Y= np.zeros(10)
    for node_i in range(node_num):
        node_data = indices_each_node[node_i]
        for label_index in node_data:
            label = train_label_orig[label_index]
            Y[label] += 1

    #距离计算结果列表
    dis=[0,0]
    for node_i in range(node_num):
        # 统计各个节点上的不同label个数
        node_data = indices_each_node[node_i]
        y_list= np.zeros(10)
        for label_index in node_data:
            label=train_label_orig[label_index]
            #print(node_i,label_index,label)
            y_list[label]+=1

        #节点与总体分布差异的距离存入列表
        #欧式（Euc），标准欧式（Sde），麦哈顿（Manh），切比雪夫（Chb），闵可夫斯基（Mik），马氏（Mab）

        if dis_type is not None:
            if dis_type=='Euc':
                distance = Euc_distance(y_list, Y)
            elif dis_type=='Sde':
                distance = Sde_distance(y_list, Y)
            elif dis_type == 'Manh':
                distance = Manh_distance(y_list, Y)
            elif dis_type == 'Chb':
                distance = Chb_distance(y_list, Y)
            elif dis_type == 'Mik':
                distance = Mik_distance(y_list, Y)
            elif dis_type == 'Mab':
                distance = Mab_distance(y_list, Y)
        else:
            distance = Euc_distance(y_list, Y)
        print(y_list, Y, distance)
        dis.append(distance)
    #print(sum(dis))
    sums = [dis_type, case_type, sum(dis)]
    dis[0] = dis_type
    dis[1] = case_type
    df=pd.DataFrame(columns=["dis_type","casetype","node0","node1","node2","node3","node4"])
    df.loc[len(df)+1]=dis
    df.to_csv(gl.PATH +dis_type+'_all_distance.csv', mode='a', header=False)

    # 存入距离向量的和,距离向量的模
    df=pd.DataFrame(columns=["dis_type","casetype","sum"])
    df.loc[len(df)+1]=sums
    df.to_csv(gl.PATH +dis_type+'_Case_distance.csv', mode='a', header=False)


def plot_tau_accruacy(case_type,model_name,aggre_type):
    '''
    统计不同的资源下，tau的变化情况
    :param case_type: 表示以某种方式分布的数据集
    :param model_name: 当前实验所用的模型
    :param aggre_type:
            fedavg--"avg",
            "AM",
            "DA"
    :return:
    '''
    casepath = gl.PATH +"tau/"+ case_type + "_"+aggre_type+ "_" + model_name + '_tau.csv'
    print(casepath)
    DF2 = pd.read_csv(casepath)
    colname = [column for column in DF2]
    print(colname)
    # 不同资源
    time_list = list(set(DF2['total_time'].tolist()))
    # print(time_list)
    different_time(case_type, model_name,aggre_type, time_list, DF2)

def plot_case_distance(dis_type,color,diss,isNormal=False,isColor=False,isInner=False):
    '''
    绘制不同case与总体分布的差异，用距离去量化
    :param dis_type: 距离计算method
    :param isNormal: 是否进行标准化处理后显示
    :param isColor: 是否采用不同的颜色显示
    diss:偏移量
    :return:
    '''
    if isInner:
        casepath = gl.PATH+"inner_dis/" + dis_type + '_inner_distance.csv'
        picname = './picture/distance/inner_' + dis_type + '_distance.svg'
    else:
        casepath = gl.PATH + dis_type + '_Case_distance.csv'
        picname='./picture/distance/' + dis_type + '_distance.svg'
        #标准化后的图
        if isNormal:
            picname = './picture/distance/' + dis_type + '_normal_dis.svg'
    df= pd.read_csv(casepath)
    colname = [column for column in df]
    print(colname)
    #距离数据
    data=df['distance'].tolist()
    if isNormal:
        maxi=max(data)
        mini=min(data)
        for i in range(len(data)):
            print(data[i])
            data[i]=(data[i]-mini)/(maxi-mini)+diss
    Xaxis = 'different case'
    Yaxis = dis_type + ' distance'

    plt.xlabel(Xaxis)
    plt.ylabel(Yaxis)
    title = 'case distibution distance'
    plt.title(title, fontsize=20)
    plt.bar(df['casetype'].tolist(), data,color=color)
    plt.savefig(picname, format='svg')
    plt.show()
    if isColor:
        color_list = ["blue", "green", "red"]
        Y_list=data
        plot_bar_color_value(Y_list, color_list, Xaxis, Yaxis, title, ymax=1.5)






def plot_bar_color_value(Y_list,color_list,Xaxis,Yaxis,title,ymax):
    '''
    绘制不同颜色的柱状图，包含数值
    Y_list:包含每个柱子的value，list
    color_list:每个柱子的颜色,str list
    Xaxis:X轴的名字,str
    Yaxis:Y轴的名字,str
    title:图的名字
    ymax:y轴的最大值
    :return:
    '''
    #x_list存放每个柱子的中心位置下标
    x_list = []
    #柱子的个数
    num = 3
    #柱子的宽度
    width = 1
    sum = width * num
    for i in range(0, sum, width):
        x_list.append(i + width / 2)
    print(x_list)
    #存放每个柱子的数值zip
    zip_list=[]
    #绘制柱子
    for i in range(len(Y_list)):
        plt.bar([x_list[i]], [Y_list[i]], 1, color=color_list[i])
        zip_list.append(zip([x_list[i]], [Y_list[i]]))
    plt.xlabel(Xaxis)  # 设置X轴Y轴名称
    plt.ylabel(Yaxis)
    plt.title(title)
    #%.2f :显示浮点数，%.0f:显示舍去小数位的整数
    for zi in zip_list:
        for a, b in zi:
            plt.text(a, b + 0.05, '%.2f'%b, ha='center', va='bottom', fontsize=11)
    plt.ylim(0, ymax)  # 设置Y轴上下限
    plt.show()