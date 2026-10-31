"""
案例：
    演示CNN的综合案例：图像分类

回顾深度学习的步骤：
    1.数据准备
     这里我们用的时候是用的计算机视觉模块torchvision 自带的CIFAR10数据集，包括了60000张32x32的RGB图片，分为10个类别，每个类别有6000张图片
     所以需要自己安装一下torchvision数据包，即pip install torchvision
    2.搭建（卷积）神经网络
    3.模型训练
    4.模型测试

卷积层：
    用于提取图像的局部特征，结合卷积核（每个卷积核=1个神经元）实现，处理后的结果叫：特征图（Feature Map）
    计算方式：N= (W-K+2P)/S + 1
    每一个卷积都是一个神经元
池化层：
    降维，有最大池化和平均池化两种
    池化只在HW上做调整，通道上不改变

案例的优化思路：
    1.增加卷积核的输出通道数
    2.增加全连接层的参数量
    3.调整学习率
    4.调整优化方案
    5.调整激活函数
    ....
"""
#导包
import torch
import torch.nn as nn
from numpy.ma.core import argmax
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor  # pip install torchvision -i https://mirrors.aliyun.com/pypi/simple/
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from torchsummary import summary

# 每批次样本数
BATCH_SIZE = 8

#1.准备数据集
def create_dataset():
    # 1.1获取训练集
    train_dataset = CIFAR10(root='data', train=True, transform=ToTensor(), download=True)
    # 1.2获取测试集
    test_dataset = CIFAR10(root='data', train=False, transform=ToTensor(), download=True)
    return train_dataset, test_dataset

#2.搭建（卷积）神经网络
class ImageModel(nn.Module):
    #2.1初始化父类成员，搭建神经网络
    def __init__(self):
        #1。初始化父类成员
        super().__init__()
        #2.搭建神经网络
        #第一个卷积层,输入的参数为：输入3通道数，输出6通道数，卷积核大小3，步幅1，padding 0
        self.conv1=nn.Conv2d(3,6,3,1,0)
        #第一个池化层，输入的参数为：池化核大小2，步幅2，padding 0
        self.pool1=nn.MaxPool2d(2,2,0)
        #第二个卷积层,输入的参数为：输入6通道数，输出16通道数，卷积核大小3，步幅1，padding 0
        self.conv2=nn.Conv2d(6,16,3,1,0)
        #第二个池化层，输入的参数为：池化核大小2，步幅2，padding 0
        self.pool2=nn.MaxPool2d(2,2,0)

        #第一个隐藏层:全连接层，输入
        self.linear1=nn.Linear(576,120)
        #第二个隐藏层:全连接层，输入120，输出84
        self.linear2=nn.Linear(120,84)
        #输出层:全连接层，输入84，输出10
        self.output=nn.Linear(84,10)
    #2.2 定义前向传播
    def forward(self, x):
        #第一层：卷积层（加权求和）+激励层（激励函数）+池化层（降维）
        # #分解写法：
        # x=self.conv1(x)
        # x=self.relu(x)
        # x=self.pool1(x)
        #合并版写法：池化+激励+卷积
        x=self.pool1(torch.relu(self.conv1(x)))
        #第二层：卷积层（加权求和）+激励层（激励函数）+池化层（降维）
        x=self.pool2(torch.relu(self.conv2(x)))

        # 第二次池化后为 (B,16,6,6)，按样本展平为 (B,576)，供本例全连接层使用
        # x.size(0) 是当前批量大小，-1 自动计算为 16*6*6=576
        x=x.reshape(x.size(0),-1)

        #第三层：全连接层（加权求和）+激励层（激励函数）
        x=torch.relu(self.linear1(x))
        #第四层：全连接层（加权求和）+激励层（激励函数）
        x=torch.relu(self.linear2(x))
        #第五层：输出层（加权求和）+激励层（激励函数）
        x=self.output(x)#输出层，因为是分类问题，所以输出层不加激励函数，后面会用多分类交叉熵损失函数=CrossEntropyLoss()
        return x

#3.模型训练
def train(train_dataset):
    #1.创建数据加载器
    dataloader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
    #2.创建模型对象
    model=ImageModel()
    #3.创建损失函数对象
    criterion=nn.CrossEntropyLoss()
    #4.创建优化器对象
    optimizer=optim.SGD(model.parameters(),lr=1e-3)
    #5.循环遍历epoch，开始每轮的训练
    #5.1定义变量，记录训练的轮数
    epochs=100
    #5.2遍历，完成每轮的所有轮次的动作
    for epoch_idx in range(epochs):
        #5.2.1定义变量，记录：总损失，总样本数据量，预测正确样本个数，训练开始时间
        total_loss, total_samples, total_correct, start_time = 0, 0, 0, time.time()
        #5.2.2遍历数据加载器，完成每轮的训练，从而获取每批次的数据
        for x,y in dataloader:
            #5.2.3切换训练模式
            model.train()
            #5.2.4前向传播，获取预测结果
            y_pre=model(x)
            #5.2.5计算损失
            loss=criterion(y_pre,y)
            #5.2.6梯度清零+反向传播+参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # #5.2.7统计预测正确的样本个数
            # print(y_pre)
            # argmax 返回最大类别分数对应的索引，作为预测类别
            # print(torch.argmax(y_pre, dim=1))#-1表示最后一个维度，预测分类，返回最大值对应的索引
            # print(y)
            # print(torch.argmax(y_pre, dim=1) == y)#比较预测结果和实际结果，返回布尔值
            # print(torch.argmax(y_pre, dim=1) == y).sum()#统计预测正确的样本个数
            total_correct+=(torch.argmax(y_pre, dim=1) == y).sum()
            #5.2.8 统计当前批次的总损失
            total_loss+=loss.item()*len(y)
            #5.2.9 统计当前批次的总样本数据量
            total_samples+=len(y)
        #5.2.10 走到这里说明一轮训练完成，打印该轮次的信息
        print(f'epoch:{epoch_idx + 1}, loss:{total_loss / total_samples:.2f}, '
              f'accuracy:{total_correct / total_samples:.2f}, '
              f'time:{time.time() - start_time:.2f}s')
        print('-' * 20)
        #break 这里写break表示只训练一轮，实际开发中不要写break
    #6.保存模型
    torch.save(model.state_dict(), './model/image_model.pth')

#4.模型测试
def evaluate(test_dataset):
    #1.创建测试集，数据加载器
    dataloader=DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False)
    #2.创建模型对象
    model=ImageModel()
    #3.加载模型参数
    model.load_state_dict(torch.load('./model/image_model.pth'))
    #4.定义变量统计，预测正确的样本个数，总样本个数
    total_correct, total_samples = 0, 0
    #5.遍历数据加载器，获取到每批次的数据
    for x,y in dataloader:
        #5.1切换模型
        model.eval()
        #5.2前向传播，获取预测结果
        y_pre=model(x)
        #5.3统计预测正确的样本个数，因为训练的时候用了CrossEntropyLoss()，所以搭建网络的时候输出层不加softmax激励函数
        # argmax 返回最大类别分数对应的索引，作为预测类别
        y_pre=torch.argmax(y_pre, dim=-1)#-1这里表示最后一个维度，预测分类，返回最大值对应的索引
        #5.4统计预测正确的样本个数
        total_correct+=(y_pre==y).sum()
        #5.5统计总样本个数
        total_samples+=len(y)
    #5.5打印测试结果
    print(f'test accuracy:{total_correct / total_samples:.2f}')

#5.测试
if __name__ == '__main__':
    #1.获取数据集
    train_dataset, test_dataset = create_dataset()
    # print(f'训练集:{train_dataset.data.shape}')# (50000, 32, 32, 3)
    # print(f'测试集:{test_dataset.data.shape}')# (10000, 32, 32, 3)
    # #{'airplane':0, 'automobile':1, 'bird':2, 'cat':3, 'deer':4, 'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}
    # print(f'数据集类别：{train_dataset.class_to_idx}')
    #
    # #图像展示
    # plt.figure(figsize=(2,2))
    # plt.imshow(train_dataset.data[11])#获取索引为11也就是第12张图片
    # plt.title(train_dataset.targets[11])
    # plt.show()
    # #2.搭建神经网络
    # model=ImageModel()
    # #查看模型参数
    # summary(model,(3,32,32),batch_size=BATCH_SIZE,device='cpu')
    # #3模型训练
    train(train_dataset)
    #4.模型测试
    evaluate(test_dataset)