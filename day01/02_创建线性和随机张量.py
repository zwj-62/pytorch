"""
创建线性和随机张量
案例：
    演示PyTorch 中如何创建线性和随机张量
涉及到的函数有：
    torch.arange()和torch.linspace()创建线性张量
    torch.random.initial_seed()和torch.random.manual_seed()随机种子设置
    torch.rand torch.randn 创建随机浮点型张量
    torch.randint（low,high, size=()) 创建随机整数型张量
需要掌握的函数有：
    arange().linspace(),manual_seed(),randint()

"""
#导包

import torch

#1.定义函数，演示创建线性张量
def dm01():
    #场景一：创建指定范围的线性张量
    t1=torch.arange(0,10,2)#包左不包右
    print(f't1:{t1},type:{type(t1)}')
    print('-'*40)
    #场景二：创建指定范围的线性张量--->等差数列
    t2=torch.linspace(1,10,4)
    print(f't2:{t2},type:{type(t2)}')
    print('-'*40)
#2.定义函数，演示创建随机张量
def dm02():
    #step1：设置随机种子，往后所有的随机用这个
    # torch.initial_seed()#默认采用系统的时间戳作为随机种子，每次运行结果都不同
    torch.manual_seed(10)#设置随机种子，每次得到的结果都一样
    #step2：创建随机张量
    #场景一：均匀分布的（0，1）张量
    t1=torch.rand(size=(2,3))
    print(f't1:{t1},type:{type(t1)}')
    print('-'*40)
    #场景二：符合正态分布的随机张量
    t2=torch.randn(size=(2,3))
    print(f't2:{t2},type:{type(t2)}')
    print('-'*40)
    #场景三：创建随机整数张量
    t3=torch.randint(1,10,size=(3,4))
    print(f't3:{t3},type:{type(t3)}')
    print('-'*40)
#3.测试函数
if __name__ == '__main__':
    dm01()
    dm02()
