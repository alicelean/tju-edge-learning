from matplotlib import pyplot as plt
from util.distance import *
import pandas as pd
from util.utils import *
import result_value.value as gl
from data_reader.data_reader import *
from config import *
def guiyihua(y):
    '''
    将向量y进行归一化处理
    :param y:
    :return:
    '''
    s=0.0
    for i in y:
        s+=float(i)
    re=[]
    for i in y:
        re.append(i/s)
    return re




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

def plot_line2(data,j,case_type=None,xyran=np.arange(0, 6000, 100)):
    '''
    :param data: 要描绘的数据
    :param j:
    :return:
    '''
    #data.sort()
    plt.xlabel("picture label")
    plt.ylabel('label num')
    # plt.yticks(xyran)
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
        plt.yticks(np.arange(0.6, 1.0, 0.05))
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
    if case_type == "case4":
        indices_each_node = get_case_4(n_nodes, train_label_orig)
    if case_type == "case5":
        indices_each_node = get_case_5(n_nodes, train_label_orig)
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
    print("case_type is ",case_type)
    if case_type == "case1":
        indices_each_node = get_case_1(n_nodes, train_label_orig)
    if case_type == "case2":
        indices_each_node = get_case_2(n_nodes, train_label_orig)
    if case_type == "case3":
        indices_each_node = get_case_3(n_nodes, train_label_orig)
    if case_type == "case4":
        indices_each_node = get_case_4(n_nodes, train_label_orig)
    if case_type == "case5":
        indices_each_node = get_case_5(n_nodes, train_label_orig)
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
    nodedf = pd.DataFrame(columns=["case","node","node1_dis", "node2_dis", "node3_dis", "node4_dis", "node5_dis"])
    for i in range(len(node_label_list)):
        Y=node_label_list[i]
        #记录节点与其他节点之间的distance
        node_distance = [case_type,i]
        for y_list in node_label_list:
            distance=alldistance(y_list, Y, dis_type)
            node_distance.append(distance)
            distance_list.append(distance)
        nodedf.loc[len(nodedf)+1]=node_distance
    nodedf.to_csv(gl.PATH + "inner_dis/" + dis_type + '_each_dis.csv', mode='a', header=False)

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
    if case_type == "case4":
        indices_each_node = get_case_4(n_nodes, train_label_orig)
    if case_type == "case5":
        indices_each_node = get_case_5(n_nodes, train_label_orig)
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
            elif dis_type == 'Kl':
                distance = KL_divergence(y_list, Y)
            elif dis_type == 'Js':
                distance = JS_divergence(y_list, Y)

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
    df.to_csv(gl.PATH+"distance/"+dis_type+'_all_distance.csv', mode='a', header=False)

    # 存入距离向量的和,距离向量的模
    df=pd.DataFrame(columns=["dis_type","casetype","sum"])
    df.loc[len(df)+1]=sums
    df.to_csv(gl.PATH +"distance/"+dis_type+'_Case_distance.csv', mode='a', header=False)


def get_wegiht(dis_type,case_type):
    path=gl.PATH + "distance/" + dis_type + '_all_distance.csv'
    df=pd.read_csv(path)
    data=df[df['case']==case_type][['node1', 'node2', 'node3', 'node4', 'node5']]

    dlist=data.values.tolist()[0]

    #求和
    s = 0.0
    for i in dlist:
        s=s+float(i)
    dlist = [float(s - i) for i in dlist]
    # 求和
    s = 0.0
    for i in dlist:
        s = s + float(i)
    print(s)
    w=[float(i)/float(s) for i in dlist]
    print(case_type, dis_type, w)
    return w


def get_wegiht2(dis_type,case_type):
    path=gl.PATH + "distance/" + dis_type + '_all_distance.csv'
    print("read ",path)
    df=pd.read_csv(path)
    data=df[df['case']==case_type][['node1', 'node2', 'node3', 'node4', 'node5']]
    dlist=data.values.tolist()[0]
    print("orgin dis is ",dlist)
    dlist = guiyihua(dlist)
    print("guiyihua dis is ", dlist)
    #求和
    w= [1.0/i for i in dlist]
    #是否进行归一化
    # w=guiyihua(w)
    # w = [1-i for i in dlist]
    print(case_type, dis_type, w)
    return w


def plot_tau_accruacy(case_type,model_name,aggre_type,mo_name):
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
    casepath = gl.PATH +"tau/"+mo_name+ case_type + "_"+aggre_type+ "_" + model_name + '_tau.csv'
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
        casepath = gl.PATH +'distance/'+ dis_type + '_Case_distance.csv'
        picname='./picture/distance/' + dis_type + '_distance.svg'
        #标准化后的图
        if isNormal:
            picname = './picture/distance/' + dis_type + '_normal_dis.svg'
    df= pd.read_csv(casepath)
    colname = [column for column in df]
    #num,method,casetype,distance
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
    plt.yticks(np.arange(0.6, 1.0, 0.05))
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




def get_node_inner_distance(case_type,dis_type=None):
    '''
    绘制某个数据的分布的各个节点的内在差异
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



def plot_multiline(aggre_type,case_type,x,y1,y2,label1,label2,title):
    '''
    :param x: x轴标签列表
    :param y1: x轴对应的数据y1
    :param y2: x轴对应的数据y2
    :param label1: y1曲线的名称
    :param label2: y2曲线的名称
    title:图表的名称
    :return:
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
    plt.title(title)
    #设置字体
    #my_font = fm.FontProperties(fname="/usr/share/fonts/wqy-microhei/wqy-microhei.ttc")
    #设置每一条曲线的样式，颜色，形状，宽度，图例信息
    ln1, = plt.plot(x, y1, color='red', linewidth=2.0, linestyle='--')
    ln2, = plt.plot(x, y2, color='blue', linewidth=3.0, linestyle='-.')
    plt.yticks(np.arange(0.8,0.9,0.02))
    plt.legend(handles=[ln1, ln2], labels=[label1, label2])
    #设置边框信息
    ax = plt.gca()
    ax.spines['right'].set_color('none')  # right边框属性设置为none 不显示
    ax.spines['top'].set_color('none')  # top边框属性设置为none 不显示
    plt.savefig('./picture/compare/'+aggre_type+"/"+ aggre_type+"_"+case_type+"_"+model_name + '.svg', format='svg')
    plt.show()


def plot_listline(path,x,ylist,labellist,colist,title,xyran):
    '''
    :param x: x轴标签列表
    :param ylist: x轴对应的多条数据y
    :param labellist: x轴对应的多条数据y的曲线的名称
    :param title:图表的名称
    :return:
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
    plt.title(title)
    #设置字体
    #my_font = fm.FontProperties(fname="/usr/share/fonts/wqy-microhei/wqy-microhei.ttc")
    #设置每一条曲线的样式，颜色，形状，宽度，图例信息
    lnlist=[]
    for i in range(len(ylist)):
        print(len(x))
        print(len(ylist[i]))
        print(len(colist[i]))
        if i ==0:

            ln, = plt.plot(x, ylist[i], color=colist[i], linewidth=2.0, linestyle='--')
        else:
            ln, = plt.plot(x, ylist[i], color=colist[i], linewidth=3.0, linestyle='-.')
        lnlist.append(ln)
    # plt.yticks(np.arange(0.75,0.95,0.05))
    plt.yticks(xyran)
    plt.legend(handles=lnlist, labels=labellist)
    #设置边框信息
    ax = plt.gca()
    ax.spines['right'].set_color('none')  # right边框属性设置为none 不显示
    ax.spines['top'].set_color('none')  # top边框属性设置为none 不显示
    plt.savefig(path,format='svg')

    plt.show()


def plot_difference_reulst(model_name,case_type,aggre_type,mo_name):
    '''
    绘制不同模型,联邦学习和集中式学习下的差距
    :return:
    '''
    #文件所在位置
    centerpath=gl.PATH+"center/"+model_name+".csv"
    fedpath= gl.PATH +"tau/"+mo_name+ case_type + "_"+aggre_type+ "_" + model_name + '_tau.csv'
    DF1= pd.read_csv(centerpath)
    DF2 = pd.read_csv(fedpath)
    # colname1 = [column for column in DF1]
    # colname2 = [column for column in DF2]
    #获取不同资源
    time_list1 = set(DF1['total_time'].tolist())
    time_list2 = set(DF2['total_time'].tolist())
    time_list=list(time_list1&time_list2)
    data={'aggre': aggre_type,"case":case_type}
    data["time" ] = time_list
    data_list=[]
    # 取指定资源数据
    for total_time in time_list:
        title = case_type + ": total_time is " + str(total_time) + " by " + model_name+"(aggregator="+aggre_type+")"
        #集中式训练结果
        cendata=DF1[DF1['total_time']==total_time][['tau', 'accuracy']]
        #联邦训练结果
        feddata= DF2[DF2['total_time'] == total_time][['tau', 'accuracy']]
        # print(cendata)
        # print(feddata)
        label1="centerlized learning"
        label2="fedrated learning"
        # plt.title(title, fontsize=20)
        #s.sort_index(axis=0,by='tau',ascending=True)
        plt.xlabel('each node local model update times')
        plt.ylabel('global model accuracy '+aggre_type)
        x=feddata['tau'].tolist()
        cen_y=cendata['accuracy'].tolist()
        fed_y=feddata['accuracy'].tolist()

        for i in range(len(fed_y)-1):
            cen_y.append(cen_y[0])
        #print(fed_y)
        # print(x)
        # print(cen_y)
        data_list.append([x, cen_y, fed_y, label1, label2, title])
        plot_multiline(aggre_type,case_type,x, cen_y, fed_y, label1, label2, title)
        #data.append([total_time,x, cen_y, fed_y, label1, label2, title])
    data["data"] =data_list
    return data


        # plt.savefig('./picture/tau_result/' + case_type + "_" + model_name + '.svg', format='svg')
        # plot_two(x,y)

def get_inner_dis_wegiht(case_type,dis_type):
    '''
    计算节点内部差异导致的权重
    :param case_type:
    :param dis_type:
    :return:
    '''
    path=gl.PATH + "inner_dis/" + dis_type + '_each_dis.csv'
    df=pd.read_csv(path)
    casedf=df[df['case']==case_type]
    inner_dis=[]
    s=0
    for i in range(len(casedf)):
        colname='node'+str(i+1)+'_dis'
        value=sum(casedf[colname].tolist())
        s+=value
        inner_dis.append(value)
    inner_dis=[s-i for i in inner_dis]
    inner_dis=guiyihua(inner_dis)
    print(case_type,inner_dis)
    return inner_dis



def plot_compare(case_list,model_name,aggre_type_list,mo_name):
    for aggre_type in aggre_type_list:
        x=None
        cen_y=None
        ylist=[cen_y]
        labellist=["center"]
        title="global accuracy trend compare (aggregator="+aggre_type+")"
        for case_type in case_list:
            labellist.append(case_type)
            data=plot_difference_reulst(model_name,case_type,aggre_type,mo_name)
            #不同的total_time 对应的数据：
            #不同资源(total_time)个数
            num_time=len(data['time'])
            for i in range(num_time):
                time_data=data['data'][i]
                total_time=data['time'][i]
                #print(i,total_time,len(time_data))
                #(x, cen_y, fed_y, label1, label2, title)
                if i ==0:
                    x=time_data[0]
                    cen_y=time_data[1]
                # all_data[aggre_type][case_type][total_time]=time_data
                ylist.append(time_data[2])
        # print(x)
        ylist[0]=cen_y
        #print("ylist",len(ylist),len(labellist))
        colist=["red","blue","green","black"]
        savpath = './picture/compare/' + aggre_type + "/" + aggre_type + "_" + model_name + '.svg'
        xyran = np.arange(0.1, 1.0, 0.05)
        plot_listline(savpath,x, ylist, labellist,colist, title,xyran)


def get_time_list(case_list,aggre_type_list,mo_name):
    #从文件中获取实验所使用的time,返回列表
    time_list = []
    for case_type in case_list:
        for aggre_type in aggre_type_list:
            # print(case_type, aggre_type)
            #从文件中提取两列数据
            path=gl.PATH +"tau/"+mo_name+ case_type + "_"+aggre_type+"_" + model_name + '_tau.csv'
            df=pd.read_csv(path)
            try:
                for i in list(set(df['total_time'].tolist())):
                    time_list.append(i)
            except:
                print(df.columns)
                print(path)
    time_list=list(set(time_list))
    return time_list

def get_case_accuracy(case_type,aggre_type,time,mo_name):
    '''
    获取指定case下，指定聚合方法下，不同资源time_list的模型准确率数据
    :param case_type:
    :param aggre_type:
    :param time_list: 不同资源
    :return:[time,x,acc]
    '''
    path = gl.PATH + "tau/" +mo_name+ case_type + "_" + aggre_type + "_" + model_name + '_tau.csv'
    df = pd.read_csv(path)
    # 从文件中提取两列数据
    acc=df[df['total_time']==time]['accuracy'].tolist()
    x=df[df['total_time']==time]['tau'].tolist()
    return x,acc

def plot_multicompare_of_case(time_list,case_list,aggre_type_list,colist,mo_name):
    '''
    绘制指定资源下，特定case下，不同aggregator的对比图（tau&accuracy）
    :param time_list:
    :param case_list:
    :param aggre_type_list:
    :return:
    '''
    print("start plot_multicompare_of_case")
    label = []
    title = []
    accuracy = []
    x = None
    #数据准备
    for time in time_list:
        # re[time]={}
        # 获取集中训练数据
        path = gl.PATH + "center/" + model_name + '.csv'
        print("print picture with time=", time, "path is :", path)
        cdf = pd.read_csv(path)
        #取出对应资源集中式训练的accuracy
        cacc = cdf[cdf['total_time'] == time]['accuracy'].tolist()
        print("center training accuracy length is ",len(cacc))
        #对每一个case下的结果数据都要进行绘制，title,每一条先的label
        for case_type in case_list:
            title.append("accuracy trend with different aggregator"+"_"+str(time)+" by "+case_type)
            # re[time][case_type]={}
            label_list = []
            acc_list = []
            cenacc = []
            for aggre_type in aggre_type_list:
                x,acc=get_case_accuracy(case_type,aggre_type,time,mo_name)
                print(case_type,aggre_type, "acc length is:", len(acc))
                label_list.append(aggre_type + "_acc")
                acc_list.append(acc)
                # re[time][case_type][aggre_type] = acc
            for i in range(len(acc_list[0])):
                cenacc.append(cacc)
            print(case_type,"center training accuracy length is ",len(cenacc))
            label_list.append("center_acc")
            acc_list.append(cenacc)

            label.append(label_list)
            accuracy.append(acc_list)

    #绘制图片，一个case一个图片
    for i in range(len(title)):
        casetit=title[i]
        labellist=label[i]
        print("case info :",casetit,labellist,len(accuracy[i]))
        path='./picture/aggregator/'+ model_name + '.svg'
        xyran = np.arange(0.73, 0.9, 0.05)
        plot_listline(path, x, accuracy[i], labellist, colist, casetit,xyran)



