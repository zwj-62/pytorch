"""
案例：
    绘制激活relu函数图像
relu 激活函数介绍：
    激活函数的目的：
    给模型增加非线性功能，让模型（神经玩）既可以做分类，还可以做线性回归
激活函数分类：
    relu
    relu
    ReLU
    Softmax
relu 激活函数：
    主要应用于二分类的输出层，且适用于 浅层神经网络（不超过5层）
    数据在[-6,6]之间有效果，在此范围外效果很差，在[-3,3]之间效果最明显
    求导后范围在[0,0.25]之间
    relu函数公式：1/(1+e^-x)
relu:
    主要应用于隐藏层，且适用于浅层神经网络（不超过5层）
    数据在[-3，3]之间有效果，在此范围外效果很差，在[-1,1]之间效果最明显
    求导后范围在[0,1]之间
    relu函数公式：(e^x-e^-x)/(e^x+e^-x)
ReLU:
    计算公式为：max(0,x)，计算量相对较小，训练成本低i，多应用于隐藏层，且适合深层神经网络
    求导后要么是0要么是1，相较于relu，收敛速度更快
    默认情况下ReLU只考虑正样本，即负样本的导数为0
Softmax:
    将多分类的结果以概率的形式展示，且概率相加为1，最终选取概率最大的分类作为最终结果
细节：
    绘制激活函数图像时出现以下提示：需要将anaconda3等
记忆：如何选择激活函数
    1.输出层：二分类：Sigmoid，多分类：softmax.回归问题：identity
    2.隐藏层：relu>Leaky RelU>tanh>sigmoid


"""
#导包
import matplotlib.pyplot as plt
import numpy as np
import torch
from sympy.stats.rv import probability

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
#1.定义张量，记录分类数据
scores=torch.tensor([[3.0,1.0,0.2,0.1,0.05,3.75],[0.1,0.02,0.05,0.1,0.2,0.3]])
#2.dim=0按行进行运算
probability=torch.softmax(scores,dim=1)
print(probability)

