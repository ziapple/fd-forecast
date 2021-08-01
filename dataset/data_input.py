from dataset import data_fea
import os
import numpy as np

data_root = "data/"
data_dir = "1st_test/"
tunnel_dir = "1st_test_tunnel/"


# 对原始数据处理,提取每个文件16个特征
def input_fea():
    c_fea = data_fea.Fea()
    for root, _, files in os.walk(os.path.join(data_root, data_dir)):
        len_files = len(files)
        index_file = 0
        # 23个特征
        for file in files:
            index_file = index_file + 1
            print("processing %s, %d/%d" % (file + ".fea", index_file, len_files))

            signals = []
            if file.endswith(".fea"):
                # os.remove(os.path.join(root, file))
                continue
            f = open(os.path.join(root, file))
            lines = f.readlines()
            for line in lines:
                params = line.strip().split("\t")
                arr_line = []
                for param in params:
                    arr_line.append(float(param))
                signals.append(arr_line)
            a = np.array(signals).T
            for i in range(a.shape[0]):
                feas = np.array(c_fea.both_fea(a[i]))
                # 一个通道生成一个文件
                f = open(os.path.join(data_root, tunnel_dir + "tunnel" + str(i) + ".fea"), "a")
                f.write(str(feas.tolist()) + "\n")
            # np.savetxt(os.path.join(root, file + ".fea"), np.array(feas_result), fmt="%.10f", delimiter=",")


if __name__ == "__main__":
    input_fea()