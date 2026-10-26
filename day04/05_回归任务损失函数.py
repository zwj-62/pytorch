"""
案例：
    回归任务损失函数

回归任务损失函数介绍如下：
    MAE: Mean Absolute Error 平均绝对误差
        公式：
            误差绝对值之和/样本总数
        类似于L1正则化，权重可以降为0，样本会变得稀疏
        弊端：
            在0点不平滑，可能出现最小值
    MSE: Mean Squared Error 平均平方误差
        公式：
            误差平方和/样本总数
        弊端：
            如果差值过大，则会出现梯度爆炸的情况

    Smooth L1: 结合MAE和MSE的特点做的升级优化，在【-1，1】范围内是MSE，在【1，无穷】范围内是MAE

        公式：
            0.5*误差平方，如果误差绝对值小于1，否则误差绝对值减去0.5
            优点：
                这样既解决了L1的0点不平滑，可能出现最小值的问题，也解决了L2的梯度爆炸问题
"""

#导包
import torch
import torch.nn as nn
#1.演示：回归任务损失函数MAE
def dm01():
    #1.1.手动创建样本的真实值
    y_true=torch.tensor([1.1,0.7,4.3],dtype=torch.float32)
    #1.2.手动创建样本的预测值
    y_pred=torch.tensor([0.1,1.4,0.9],requires_grad=True,dtype=torch.float32)
    #1.3.创建回归任务损失函数
    criterion=nn.L1Loss()
    #1.4.计算损失
    loss=criterion(y_pred,y_true)
    print(f'损失值：loss={loss}')
#2.演示：回归任务损失函数MSE
def dm02():
    #2.1.手动创建样本的真实值
    y_true=torch.tensor([1.1,0.7,4.3],dtype=torch.float32)
    #2.2.手动创建样本的预测值
    y_pred=torch.tensor([0.1,1.4,0.9],requires_grad=True,dtype=torch.float32)
    #2.3.创建回归任务损失函数
    criterion=nn.MSELoss()
    #2.4.计算损失
    loss=criterion(y_pred,y_true)
    print(f'损失值：loss={loss}')
#3.演示：回归任务损失函数Smooth L1
def dm03():

    #3.1.手动创建样本的真实值
    y_true=torch.tensor([1.1,0.7,4.3],dtype=torch.float32)
    #3.2.手动创建样本的预测值
    y_pred=torch.tensor([0.1,1.4,0.9],requires_grad=True,dtype=torch.float32)
    #3.3.创建回归任务损失函数
    criterion=nn.SmoothL1Loss()
    #3.4.计算损失
    loss=criterion(y_pred,y_true)
    print(f'损失值：loss={loss}')
if __name__=='__main__':

    dm01()
    dm02()
    dm03()