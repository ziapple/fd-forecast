## This is a pytorch implementation of the paper *[Unsupervised Domain Adaptation by Backpropagation](http://sites.skoltech.ru/compvision/projects/grl/)*


#### Environment
- Pytorch 1.0
- Python 2.7

#### Network Structure
![p8KTyD.md.jpg](https://s1.ax1x.com/2018/01/12/p8KTyD.md.jpg)

#### Dataset
First, you need download the target dataset mnist_m from [pan.baidu.com](https://pan.baidu.com/s/1J0WAKEtEinsVzny_MgldXw) 
fetch code: kbhn or

```
cd dataset
tar -zvxf IMS.7z
```

```
Set No. 1:
Recording Duration: 一个月 October 22, 2003 12:06:24 to November 25, 2003 23:39:56
No. of Files: 2,156  2156个文件
No. of Channels: 8   8个通道
Channel Arrangement: Bearing 1 – Ch 1&2; Bearing 2 – Ch 3&4; 
Bearing 3 – Ch 5&6; Bearing 4 – Ch 7&8. 
File Recording Interval: 每10分钟记录一次 Every 10 minutes (except the first 43 files were taken every 5 minutes)
File Format: ASCII
Description: 内圈故障发生在轴承3，滚珠故障发生在轴承4 At the end of the test-to-failure experiment, inner race defect occurred in 
bearing 3 and roller element defect in bearing 4.
记录格式，8个通道
-0.022	-0.039	-0.183	-0.054	-0.105	-0.134	-0.129	-0.142
-0.105	-0.017	-0.164	-0.183	-0.049	0.029	-0.115	-0.122
```

#### Training
Then, run `main.py`

python 3 and docker version please go to [DANN_py3](https://github.com/fungtion/DANN_py3) 

