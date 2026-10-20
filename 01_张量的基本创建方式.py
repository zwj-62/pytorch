"""
案例：
     演示张量的创建方式
张量：
    pytorch框架数据最常用的深度学习框架，无论是后面要学习的ANN(人工神经网络），还是CNN(卷积神经网络)，RNN（循环神经）
都是基于张量进行计算的
张量--->存储同一类型元素的容器，且元素必须是数值才可以
    创建张量的方式有种：
    torch.tensor 根据指定数据创建张量
    torch.Tensor 根据形状创建张量，也可以用来根据指定数据创建张量
    torch.IntTensor torch.FloatTensor torch.DoubleTensor 创建指定类型的张量

细节：
Tensor 相较于 tensor 的地方在于，Tensor 可以动态创建，可以直接指定数据，tensor 需要指定数据类型
需要掌握的方式：
tensor(值，类型)    例如：
    data=np.random.randint(0,10,size=(2,3))
    t3=torch.tensor(data,dtype=torch.float)
"""
#导包：
import torch
import numpy as np
#1.定义函数：torch.tensor 根据指定数据演示张量
def dm01():
    #场景一：标量 张量
    t1=torch.tensor(10)
    print(f't1:{t1},type:{type(t1)}')
    print("-"*40)

    #场景二：二维列表转为张量
    data=[[1,2,3,4],[5,6,7,8]]
    t2=torch.tensor(data)
    print(f't2:{t2},type:{type(t2)}')
    print("-"*40)

    #场景三：numpy nd数组也可以转为张量
    data=np.random.randint(0,10,size=(2,3))
    t3=torch.tensor(data,dtype=torch.float)
    print(f't3:{t3},type:{type(t3)}')
    print("-"*40)
    print('*'*40)
    #场景四，尝试直接创建张量，维度为两行三列
    # t4=torch.tensor(2,3)#报错
    # print(f't4:{t4},type:{type(t4)}')
    # print("-"*40)
#2.定义函数 ：torch.Tensor 根据形状创建张量
 #场景一：标量 张量
def dm02():
    # 场景一：标量 张量
    t1 = torch. Tensor(10)
    print(f't1:{t1},type:{type(t1)}')
    print("-" * 40)

    # 场景二：二维列表转为张量
    data = [[1, 2, 3, 4], [5, 6, 7, 8]]
    t2 = torch.Tensor(data)
    print(f't2:{t2},type:{type(t2)}')
    print("-" * 40)

    # 场景三：numpy nd数组也可以转为张量
    data = np.random.randint(0, 10, size=(2, 3))
    t3 = torch.Tensor(data)
    print(f't3:{t3},type:{type(t3)}')
    print("-" * 40)
 #场景四，尝试直接创建张量，维度为两行三列
    t4=torch.Tensor(2,3)#不会报错了，但是是填充随机数
    print(f't4:{t4},type:{type(t4)}')
    print("-"*40)
    print('*'*40)
#3.定义函数：torch.IntTensor torch.FloatTensor torch.DoubleTensor 创建指定类型的张量
def dm03():
    # 场景一：标量 张量
    t1 = torch.IntTensor(10)
    print(f't1:{t1},type:{type(t1)}')
    print("-" * 40)

    # 场景二：二维列表转为张量
    data = [[1, 2, 3, 4], [5, 6, 7, 8]]
    t2 = torch.IntTensor(data)
    print(f't2:{t2},type:{type(t2)}')
    print("-" * 40)

    # 场景三：numpy nd数组也可以转为张量1
    data = np.random.randint(0, 10, size=(2, 3))
    t3 = torch.IntTensor(data)
    print(f't3:{t3},type:{type(t3)}')
    print("-" * 40)
    # 场景四，尝试直接创建张量，维度为两行三列
    t4 = torch.IntTensor(2, 3)  # 不会报错了，但是是填充随机数
    print(f't4:{t4},type:{type(t4)}')
    print("-" * 40)
    # 场景五：如果类型不匹配，会尝试自动跳转类型
    data=np.random.randint(0,10,size=(2,3))
    t5=torch.FloatTensor(data)#默认是float32
    print(f't5:{t5},type:{type(t5)}')
    print("-" * 40)
    print('*'*40)

#4:测试函数
if __name__ == '__main__':
    dm01()#掌握
    dm02()
    dm03()