"""
案例：
    演示学习率衰减策略

学习率衰减策略介绍：
    目的：
        较于之前的AddGrade RMSProp Adam 方式，我们可以通过等间隔，指定间隔等方式，来手动控制学习率的调整
    分类：
        等间隔学习率衰减
        指定间隔学习率衰减
        指数学习率衰减

等间隔学习率衰减：
    step_size: 每隔step_size个epoch，学习率衰减一次
    gamma: 学习率衰减的幅度,lr新=lr旧*gamma
指定间隔学习率衰减：
    milestones: 指定间隔的轮数
    gamma: 学习率衰减的幅度,lr新=lr旧*gamma
指数学习率衰减：
    gamma: 学习率衰减的幅度,lr新=lr旧*gamma**epoch
    前期学习率衰减幅度较大，后期学习率衰减幅度较小
总结：
    等间隔学习率衰减：
        优点：直观，易于调试，适用于大批量数据
        缺点：
"""
#导包
import torch
from torch import optim
import matplotlib.pyplot as plt
#1.定义函数：演示等间隔学习率衰减
def dm01():
    #1.1定义变量：记录初始的学习率，训练的轮数，每轮训练的批次数
    lr,epochs,iteration=0.1,200,10
    #1.2创建数据集,y_true,x,w
    #真实值
    y_true=torch.tensor([0])
    #输入特征
    x=torch.tensor([1.0],dtype=torch.float32)
    #权重参数w,需求自动问的
    w=torch.tensor([1.0],requires_grad=True,dtype=torch.float32)
    #1.3创建优化器对象，动量法：加速模型的收敛，减少震荡
    optimizer=optim.SGD([w],lr=lr,momentum=0.9)
    #1.4创建学习率衰减对象
    #思路一：创建等间隔学习率衰减对象
    scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=50,gamma=0.5)
    #1.5创建两个列表
    lr_list,epoch_list=[],[]
    #1.6创建循环，进行训练
    for epoch in range(epochs):
        #1.7获取当前学习率和轮数放在列表中
        epoch_list.append(epoch)
        lr_list.append(scheduler.get_last_lr())#获取最后的lr，放在列表中
        #1.8循环遍历，每轮每批次进行训练
        for batch in range(iteration):
            #1.9先计算预测值，然后计算损失函数
            y_pred=w*x
            #1.10计算损失，最小二乘法
            loss=(y_pred-y_true).pow(2)
            #1.11梯度清零+计算梯度+更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #1.12更新学习率
        scheduler.step()
    #1.13绘制学习变化曲线
    plt.plot(epoch_list,lr_list,label='Step LR Scheduler')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.show()
#2.定义函数：演示指定间隔学习率衰减
def dm02():
    #2.1定义变量：记录初始的学习率，训练的轮数，每轮训练的批次数
    lr,epochs,iteration=0.1,200,10
    #2.2创建数据集,y_true,x,w
    #真实值
    y_true=torch.tensor([0])
    #输入特征
    x=torch.tensor([1.0],dtype=torch.float32)
    #权重参数w,需求自动问的
    w=torch.tensor([1.0],requires_grad=True,dtype=torch.float32)
    #2.3创建优化器对象，动量法：加速模型的收敛，减少震荡
    optimizer=optim.SGD([w],lr=lr,momentum=0.9)
    #2.4创建学习率衰减对象
    # #思路一：创建等间隔学习率衰减对象
    # scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=50,gamma=0.5)
    #思路二：创建指定间隔学习率衰减对象
    #定义变量，记录需要修改学习率的轮数
    milestones=[50,120,160]
    #创建指定间隔学习率衰减对象
    scheduler=optim.lr_scheduler.MultiStepLR(optimizer,milestones=milestones,gamma=0.5)
    #2.5创建两个列表
    lr_list,epoch_list=[],[]
    #2.6创建循环，进行训练
    for epoch in range(epochs):
        #2.7获取当前学习率和轮数放在列表中
        epoch_list.append(epoch)
        lr_list.append(scheduler.get_last_lr())#获取最后的lr，放在列表中
        #2.8循环遍历，每轮每批次进行训练
        for batch in range(iteration):
            #2.9先计算预测值，然后计算损失函数
            y_pred=w*x
            #2.10计算损失，最小二乘法
            loss=(y_pred-y_true).pow(2)
            #2.11梯度清零+计算梯度+更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #2.12更新学习率
        scheduler.step()
    #2.13绘制学习变化曲线
    plt.plot(epoch_list,lr_list,label='Step LR Scheduler')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.show()
#3.定义函数：演示指数学习率衰减
#2.定义函数：演示指定间隔学习率衰减
def dm03():
    #3.1定义变量：记录初始的学习率，训练的轮数，每轮训练的批次数
    lr,epochs,iteration=0.1,200,10
    #3.2创建数据集,y_true,x,w
    #真实值
    y_true=torch.tensor([0])
    #输入特征
    x=torch.tensor([1.0],dtype=torch.float32)
    #权重参数w,需求自动问的
    w=torch.tensor([1.0],requires_grad=True,dtype=torch.float32)
    #3.3创建优化器对象，动量法：加速模型的收敛，减少震荡
    optimizer=optim.SGD([w],lr=lr,momentum=0.9)
    #3.4创建学习率衰减对象
    # #思路一：创建等间隔学习率衰减对象
    # scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=50,gamma=0.5)
    # #思路二：创建指定间隔学习率衰减对象
    # #定义变量，记录需要修改学习率的轮数
    # milestones=[50,120,160]
    # #创建指定间隔学习率衰减对象
    # scheduler=optim.lr_scheduler.MultiStepLR(optimizer,milestones=milestones,gamma=0.5)
    #思路三：创建指数学习率衰减对象
    scheduler=optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.95)
    #3.5创建两个列表
    lr_list,epoch_list=[],[]
    #3.6创建循环，进行训练
    for epoch in range(epochs):
        #3.7获取当前学习率和轮数放在列表中
        epoch_list.append(epoch)
        lr_list.append(scheduler.get_last_lr())#获取最后的lr，放在列表中
        #3.8循环遍历，每轮每批次进行训练
        for batch in range(iteration):
            #3.9先计算预测值，然后计算损失函数
            y_pred=w*x
            #3.10计算损失，最小二乘法
            loss=(y_pred-y_true).pow(2)
            #3.11梯度清零+计算梯度+更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #3.12更新学习率
        scheduler.step()
    #3.13绘制学习变化曲线
    plt.plot(epoch_list,lr_list,label='Step LR Scheduler')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.show()
#测试函数
if __name__ == '__main__':
    dm01()
    dm02()
    dm03()
