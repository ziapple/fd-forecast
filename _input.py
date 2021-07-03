from scipy.io import loadmat
import numpy as np
import random
import os
from keras.utils import to_categorical
from scipy.fftpack import fft
import pywt
from PyEMD import EEMD

# 采样频率
TIME_PERIODS = 6000
# 标签类别
LABEL_SIZE = 10
# Data目录
DATA_DIR = "./data/1st_test_sample/"


# Data下面包含三组数据，每组数据记录了从正常到失败实验数据，每组数据很多文件组成，每10分钟采集一次，采集一次生成一个文件
# 每个文件代表1秒采集的振动信号，每个文件包含20480个采集点（采样率为20HZ），文件名用采集的时间命名；
# 文件每行代表一组采集点集合，采集点集合由NI DAQ Card 6062E传感器采集；
# ----------------
# Set No. 2:
# Recording Duration: October 22, 2003 12:06:24 to November 25, 2003 23:39:56
# No. of Files: 2,156
# No. of Channels: 8
# Channel Arrangement: Bearing 1 – Ch 1&2; Bearing 2 – Ch 3&4;Bearing 3 – Ch 5&6; Bearing 4 – Ch 7&8.
# File Recording Interval: Every 10 minutes (except the first 43 files were taken every 5 minutes)
# File Format: ASCII
# Description: At the end of the test-to-failure experiment, inner race defect occurred in
# bearing 3 and roller element defect in bearing 4.
# -----------------
# Set No. 2:
# Recording Duration: February 12, 2004 10:32:39 to February 19, 2004 06:22:39
# No. of Files: 984
# No. of Channels: 4
# Channel Arrangement: Bearing 1 – Ch 1; Bearing2 – Ch 2; Bearing3 – Ch3; Bearing 4 – Ch 4.
# File Recording Interval: Every 10 minutes
# File Format: ASCII
# Description: At the end of the test-to-failure experiment, outer race failure occurred in
# bearing 1.
# -----------------
# Set No. 3
# Recording Duration: March 4, 2004 09:27:46 to April 4, 2004 19:01:57
# No. of Files: 4,448
# No. of Channels: 4
# Channel Arrangement: Bearing1 – Ch 1; Bearing2 – Ch 2; Bearing3 – Ch3; Bearing4 – Ch4;
# File Recording Interval: Every 10 minutes
# File Format: ASCII
# Description: At the end of the test-to-failure experiment, outer race failure occurred in
# bearing 3.
def read_data(x_len=TIME_PERIODS, label_size=LABEL_SIZE):
    """
    读取data下面所有文件
    :param x_len: 采样率，输入数据的长度
    :param label_size 标签大小
    :return:
    """
    x_train = np.zeros((0, x_len))
    x_test = np.zeros((0, x_len))
    y_train = []
    y_test = []
    i = 0
    # N*M数组，N代表行数,M代表每行元素个数（8个）
    time_series = []
    for item in os.listdir(DATA_DIR):
        print(item)
        file = open(DATA_DIR + item)
        for line in file.readlines():
            arr = line[0:-1].split("\t")
            arr = [float(el) for el in arr]
            time_series.append(arr)

        # 获取序列最大长度,时序数据有122571个的采集点，采样率为12K，采集了10秒,id_last=-571
        idx_last = -(time_series.shape[0] % x_len)
        # 切片分割,clips.shape=(122, 1000),从后往前取
        clips = time_series[:idx_last].reshape(-1, x_len)
        n = clips.shape[0]
        # 按3/4比例为训练数据，1/4为测试数据
        n_split = int((3 * n / 4))
        # 二维数组填充,增量式填充
        x_train = np.vstack((x_train, clips[:n_split]))
        x_test = np.vstack((x_test, clips[n_split:]))
        # [0]+[1] = [0, 1],不断累积标签
        y_train += [i] * n_split
        y_test += [i] * (clips.shape[0] - n_split)
        i += 1

    x_train, y_train = _shuffle(x_train, y_train)
    x_test, y_test = _shuffle(x_test, y_test)
    # y做one-hot处理
    y_train = to_categorical(y_train, label_size)
    y_test = to_categorical(y_test, label_size)

    return x_train, y_train, x_test, y_test


# 给定数组重新排列
def _shuffle(x, y):
    # shuffle training samples
    index = list(range(x.shape[0]))
    random.Random(0).shuffle(index)
    x = x[index]
    y = tuple(y[i] for i in index)
    return x, y


def x_fft(x):
    """
    傅里叶变换
    :param x 样本，每个点代表t时刻幅值
    :return: 每个点代表n频率下的幅值，由于傅里叶变换后对称性，取一半
    """
    return abs(fft(x)/2)


def x_wavelet(x):
    """
    小波变换
    :param x
    :return:
    """
    # cgau8为小波函数
    cwtmatr1, freqs1 = pywt.cwt(x_train, np.arange(1, 500), 'cgau8', 1 / 500)
    return abs(cwtmatr1)


def x_eemd(x):
    """
    eemd变换
    :param x: 代表采样数据[m,n],m代表每个周期的样本，n代表采样点
    :return: 返回每个样本IMF分量，[m,IMF,n]
    """
    eemd = EEMD()
    emd = eemd.EMD
    emd.extrema_detection = "parabol"
    for i in range(x.shape[0]):
        eIMFs = eemd.eemd(x[i])
        print(eIMFs)


if __name__ == "__main__":
    x_train, _, _, _ = read_data(x_len=400)
    # x_eemd(x_train)