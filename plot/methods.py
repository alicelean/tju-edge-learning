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

def plot_two(x,y):
    plt.plot(x, y, linewidth=3, color='red', marker='o', linestyle='--', label='我是图例')
    plt.show()