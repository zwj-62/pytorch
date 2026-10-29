"""
案例：
    ANN（人工神经网络）案例：手机价格分类案例
背景：
    基于手机的20个案例，预测手机价格区间（四个区间），可以用机器学习做也可以用深度学习做，这里用ANN来做，ANN是人工神经网络模型

ANN案例的实现步骤；
    1.构建数据集
    2.搭建神经网络
    3.模型训练
    4.模型测试
"""
#导包
import torch                                            #pytorch 框架，封装了张量的各种操作
from torch.utils.data import TensorDataset              #数据集对象，数据---Tensor ---数据库---数据加载器
from torch.utils.data import DataLoader                 #数据加载器
from torch import nn                                    #neural network 封装了神经网络的各种操作
from torch import optim                                 #优化器
from sklearn.model_selection import train_test_split    #训练集和测试集的划分
import matplotlib.pyplot as plt                         #绘图
import numpy as np                                      #数组/矩阵操作
import pandas as pd                                     #数据处理
import time                                             #时间模块

#todo 1.定义函数，构建数据集
def create_dataset():
    #1.加载csv数据D:\pytorch\day05\data\手机价格预测.csv
    data=pd.read_csv(r'/day06/data\手机价格预测.csv')
    # print(f'data:{data.head()}')
    # print(f'data.shape:{data.shape}')#2000 ,21
    #2.获取x的特征列和y的标签列
    x,y=data.iloc[:,:-1],data.iloc[:,-1]
    # print(f'x.shape:{x.shape},y.shape:{y.shape}')
    #3.把特征列转换成浮点数
    x=x.astype(np.float32)
    # print(f'x:{x.head()}')
    #4.把数据集划分为训练集和测试集
    #参数一：特征，参数二：标签，参数三：测试集比例，参数四：随机种子，参数五：分层采样
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=3,stratify=y)
    #5.把数据集封装成张量数据集，思路：数据---张量Tensor---数据集TensorDataSet---数据加载器DataLoader
    train_dataset=TensorDataset(torch.tensor(x_train.values),torch.tensor(y_train.values))
    test_dataset=TensorDataset(torch.tensor(x_test.values),torch.tensor(y_test.values))
    # print(f'train_dataset:{train_dataset}')
    # print(f'test_dataset:{test_dataset}')
    #x_train_shape[1]是特征的维度20，len(np.unique(y))是标签的类别数4
    return train_dataset,test_dataset,x_train.shape[1],len(np.unique(y))

#todo 2.构建神经网络
#todo 3.模型训练
#todo 4.模型测试
#todo 5.测试
if __name__ == '__main__':
    train_dataset, test_dataset, input_dim, output_dim = create_dataset()
    print(f'训练集数据集对象：{train_dataset}')
    print(f'测试集数据集对象：{test_dataset}')
    print(f'输入特征维度：{input_dim},输出标签类别数： {output_dim}')

