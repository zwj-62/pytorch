"""
案例：
    二分类任务损失函数

二分类任务损失函数：
    公式：
        Loss=-ylog(预测值）-（1-y）log(1-预测值)
    细节：
        因为公式没有包含Sigmoid函数，所以如果需要使用BCELoss，那么需要在损失计算之前手动调用Sigmoid函数

"""

#导包
import torch
import torch.nn as nn
#1.定义函数：演示：二分类任务损失函数
def d01():
    #1.手动创建样本的真实值--->就是上述公式中的y
    y_true=torch.tensor([0,1,0],requires_grad=True,dtype=torch.float32)
    #2.手动创建样本的预测值
    y_pred=torch.tensor([0.6901,0.4061,0.9851])
    #3.创建二分类任务损失函数
    criterion=nn.BCELoss()
    #4.计算损失
    loss=criterion(y_pred,y_true)
    print(f'损失值：loss={loss}')
#2.调用函数
if __name__ == '__main__':
    d01()
