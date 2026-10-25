"""
案例：
    演示如何创建全0 全1 指定值的张量
    涉及到的函数有：
    torch.ones() 和torch.ones_like 创建全1张量
    torch.zeros()和torch.zeros_like 创建全0张量
    torch.full(fill_value=一个数)和torch.full_like() 创建全为指定值张量
    需要掌握的函数有：
    zeros()，full()
"""
import torch

#导包


#场景一： torch.ones() 和torch.ones_like 创建全1张量
def dm01():
    t1 = torch.ones(2, 3)
    print(f't1:{t1},type:{type(t1)}')
    print('-' * 40)
    #创建一个三行三列的矩阵
    t2 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]] )
    #t3在t2的基础上创建一个全1张量
    t3 = torch.ones_like(t2)#创建一个三行三列的全1张量
    print(f't3:{t3},type:{type(t3)}')
    print('-' * 40)
    print('**' * 20)

# 场景二：torch.zeros() 和torch.zeros_like 创建全0张量
def dm02():
    t1 = torch.zeros(2, 3)
    print(f't1:{t1},type:{type(t1)}')
    print('-' * 40)
    #创建一个三行三列的矩阵
    t2 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]] )
    #t3在t2的基础上创建一个全0张量
    t3 = torch.zeros_like(t2)#创建一个三行三列的全0张量
    print(f't3:{t3},type:{type(t3)}')
    print('-' * 40)
    print('**' * 20)

# 场景三：torch.full()和torch.full_like创建全为指定值张量
def dm03():
    t1 = torch.full((2, 3), fill_value=5)
    print(f't1:{t1},type:{type(t1)}')
    print('-' * 40)
    #创建一个三行三列的矩阵
    t2 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]] )
    #t3在t2的基础上创建一个全255张量
    t3 = torch.full_like(t2, fill_value=255 )#创建一个三行三列的全255张量
    print(f't3:{t3},type:{type(t3)}')
    print('-' * 40)
    print('**' * 20)
if __name__ == '__main__':
    dm01()
    dm02()
    dm03()