from matplotlib import pyplot as plt
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
def different_time(case_type,model_name,time_list,DF):
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