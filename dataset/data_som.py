from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from minisom import MiniSom
from dataset import data_input
import os
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
# 训练通道
tunnel_name = "tunnel0.fea"


def load_data():
    # 加载data_input
    arr = []
    f = open(os.path.join(data_input.data_root, data_input.tunnel_dir + tunnel_name))
    for line in f.readlines():
        line = line.replace("[", "").replace("]", "")
        b = []
        [b.append(float(i)) for i in line.split(",")]
        arr.append(b)
    x_train, x_test = train_test_split(np.array(arr), test_size=0.3, random_state=0)
    # 样本数量
    global N
    global M
    global size
    N = x_train.shape[0]
    M = x_train.shape[1]  # 维度/特征数量
    size = math.ceil(np.sqrt(5 * np.sqrt(N)))  # 经验公式：决定输出层尺寸
    print("训练样本个数:{}  测试样本个数:{}".format(N, x_test.shape[0]))
    print("输出网格最佳边长为:", size)
    return x_train, x_test


def train(x_train):
    """
    第一个和第二个重要的参数是输出层的尺寸 ：我们是使用经验公式
    Neighborhood_function可选的设置有'gaussian'、'mexican_hat'、'bubble'. 调参的时候可以都试一遍，看效果
    学习率：先设为默认的0.5，大部分情况下都适用
    """
    som = MiniSom(size, size, M, sigma=3, learning_rate=0.5, neighborhood_function='bubble')
    som.pca_weights_init(x_train)
    som.train_batch(x_train, max_iter, verbose=False)
    return som


def main():
    x_train, x_test = load_data()
    som = train(x_train)
    # 测试集误差
    erros = []
    for _x in x_test:
        erros.append(som.quantization_error([_x]))
    print(erros)
    plt.plot(erros)
    plt.show()


if __name__ == "__main__":
    main()
