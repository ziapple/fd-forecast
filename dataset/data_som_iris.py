from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from minisom import MiniSom
from sklearn import datasets
import math

"""
附录：miniSomAPI参考
创建网络：创建时就会随机初始化网络权重 
som = minisom.MiniSom(size,size,Input_size, sigma=sig,learning_rate=learning_rate, neighborhood_function='gaussian')
som.random_weights_init(X_train)：随机选取样本进行初始化 som.pca_weights_init(X_train)：PCA初始化
som.get_weights()： Returns the weights of the neural network
som.distance_map()：Returns the distance map of the weights
som.activate(X)： Returns the activation map to x 值越小的神经元，表示与输入样本 越匹配
som.quantization(X)：Assigns a code book 给定一个 输入样本，找出该样本的优胜节点，然后返回该神经元的权值向量(每个元素对应一个输入单元)
som.winner(X)： 给定一个 输入样本，找出该样本的优胜节点 格式：输出平面中的位置
som.win_map(X)：将各个样本，映射到平面中对应的位置 返回一个dict { position: samples_list }
som.activation_response(X)： 返回输出平面中，各个神经元成为 winner的次数 格式为 1个二维矩阵
quantization_error(量化误差)： 输入样本 与 对应的winner神经元的weight 之间的 平方根
"""
# 样本数量， 维度/特征数量
N = 0
M = 0
max_iter = 200
# 输出尺寸
size = 0


def load_data():
    iris = datasets.load_iris()
    print('>> shape of data:',iris.data.shape)
    x = iris.data
    y = iris.target
    # 划分训练集、测试集  7:3
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    # 样本数量
    global N
    global M
    global size
    N = x_train.shape[0]
    M = x_train.shape[1]  # 维度/特征数量
    size = math.ceil(np.sqrt(5 * np.sqrt(N)))  # 经验公式：决定输出层尺寸
    print("训练样本个数:{}  测试样本个数:{}".format(N, x_test.shape[0]))
    print("输出网格最佳边长为:", size)
    return x_train, x_train, y_train, y_test


def train(x_train, y_train):
    """
    第一个和第二个重要的参数是输出层的尺寸 ：我们是使用经验公式
    Neighborhood_function可选的设置有'gaussian'、'mexican_hat'、'bubble'. 调参的时候可以都试一遍，看效果
    学习率：先设为默认的0.5，大部分情况下都适用
    """
    som = MiniSom(size, size, M, sigma=3, learning_rate=0.5, neighborhood_function='bubble')
    som.pca_weights_init(x_train)
    som.train_batch(x_train, max_iter, verbose=False)
    # 因为Som是无监督学习算法，不需要标签信息
    winmap = som.labels_map(x_train, y_train)
    return som, winmap


def classify(som, data, winmap):
    default_class = np.sum(list(winmap.values())).most_common()[0][0]
    result = []
    for d in data:
        win_position = som.winner(d)
        if win_position in winmap:
            result.append(winmap[win_position].most_common()[0][0])
        else:
            result.append(default_class)
    return result


def predict(som, x_test, winmap, y_test):
    # 输出混淆矩阵
    y_pred = classify(som, x_test, winmap)
    print(classification_report(y_test, np.array(y_pred)))


def show(som, x_train, y_train):
    # 背景上画U-Matrix
    heatmap = som.distance_map()
    plt.pcolor(heatmap, cmap='bone_r')  # plotting the distance map as background
    # 定义不同标签的图案标记
    markers = ['o', 's', 'D']
    colors = ['C0', 'C1', 'C2']
    category_color = {'setosa': 'C0', 'versicolor': 'C1', 'virginica': 'C2'}
    for cnt, xx in enumerate(x_train):
        w = som.winner(xx)  # getting the winner
        # 在样本Heat的地方画上标记+
        plt.plot(w[0] + .5, w[1] + .5, markers[y_train[cnt]], markerfacecolor='None',
                 markeredgecolor=colors[y_train[cnt]], markersize=12, markeredgewidth=2)
    plt.axis([0, size, 0, size])
    ax = plt.gca()
    ax.invert_yaxis()  # 颠倒y轴方向
    legend_elements = [Patch(facecolor=clr,
                             edgecolor='w',
                             label=l) for l, clr in category_color.items()]
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, .95))
    plt.show()


def main():
    x_train, x_test, y_train, y_test = load_data()
    som, winmap = train(x_train, y_train)
    show(som, x_train, y_train)


if __name__ == "__main__":
    main()
