"""
案例：
    多分类任务交叉熵损失函数
损失函数介绍：
    概述：
        损失函数也叫成本函数，目标函数，代价函数，误差函数，就是用来衡量模型好坏（模型拟合情况）的
    分类：
        多分类任务交叉熵损失函数：CrossEntropyLoss
        二分类任务交叉熵损失函数：BCELoss
    回归问题：
        MAE：Mean Absolute Error 平均绝对误差
        MSE：Mean Squared Error 平均平方误差
        Smooth L1:结合上述两个的特点做的升级优化

多分类任务交叉熵损失函数:CrossEntropyLoss:
    设计思路：
        Loss=-Σ[y_ilog(S(f(x))
    简单记忆：
        x:          样本
        f(x):       加权求和
        S(f(x)):    处理后的概率
        y:          样本x属于某一个类别的真实概率
        大白话解释：
            损失函数结果=确定类别概率的对数的最小化
        细节：
            CrossEntropyLoss=Softmax()+损失计算，后续如果要用到这个损失函数，那么：输出层就不用额外调用Softmax()函数了
"""
#导包
import torch
import torch.nn as nn
#1.定义函数，演示：多分类任务交叉熵损失函数
def dm01():
    #1.手动创建样本的真实值--->就是上述公式中的y
    #y_true=torch.tensor([[0,1,0],[1,0,0]])
    y_true=torch.tensor([1,2])
    #2.手动创建样本的预测值
    y_pred=torch.tensor([[0.1,0.8,0.1],[0.7,0.2,0.1]],requires_grad=True,dtype=torch.float32)
    #3.创建多分类任务交叉熵损失函数
    loss=nn.CrossEntropyLoss()
    #4.计算损失
    result=loss(y_pred,y_true)
    print(result)
#2.调用函数
if __name__ == '__main__':
    dm01()

