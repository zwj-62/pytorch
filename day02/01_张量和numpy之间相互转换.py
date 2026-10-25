"""
案例：
     演示张量和numpy之间相互转换的过程，以及如何从标量张量中提取其内容

涉及到的API：
    场景一：张量 ------>numpy nd数组对象
    张量对象.numpy()         共享内存
    张量对象.numpy().copy() 不共享内存，链式编程写法
    场景二：numpy nd数组对象 -----> 张量
    torch.from_numpy(nd数组对象)  共享内存
    torch.tensor(nd数组)         不共享内存
    场景三：从标量张量中提取其内容
    标量张量.item()

需要掌握：
    张量----->numpy :张量对象.numpy()
    numpy -----> 张量 :torch.from_numpy(nd数组对象)
    从标量张量中提取内容：标量张量.item()
"""
#导包
import torch
import numpy as np
#1.定义函数：演示：张量 ------>numpy nd数组对象
def dm01():
    #1.1创建张量对象
    t1=torch.tensor([1,2,3,4,5])
    print(f't1:{t1},type:{t1.type()}')
    #1.2张量 ------>numpy nd数组对象的两种不同方法
    nd1=t1.numpy()
    print(f'nd1:{nd1},type:{type(nd1)}')
    nd2=t1.numpy().copy()
    print(f'nd2:{nd2},type:{type(nd2)}')
    #1.3演示上述方式，内存是否共享
    t1[0] = 100
    print(f't1:{t1},type:{t1.type()}')#t1中的内容被修改
    print(f'nd1:{nd1},type:{type(nd1)}')#nd1中的内容也被修改
    print(f'nd2:{nd2},type:{type(nd2)}')


    print("-"*40)
#2.定义函数：演示：numpy nd数组对象 -----> 张量
def dm02():
    #2.1创建nd数组对象
    nd1 = np.array([11,22,33,44,55])
    print(f'nd1:{nd1},type:{type(nd1)}')
    #2.2 numpy nd数组对象 -----> 张量的两种不同方法
    t1=torch.from_numpy(nd1)
    #也可以转换类型：t1=torch.from_numpy(nd1).type(torch.float32)
    print(f't1:{t1},type:{t1.type()}')
    t2=torch.tensor(nd1)
    print(f't2:{t2},type:{t2.type()}')
    #2.3演示上述方式，内存是否共享
    t1[0] = 100
    print(f'nd1:{nd1},type:{type(nd1)}')#nd1中的内容被修改
    print(f't1:{t1},type:{t1.type()}')#t1中的内容被修改
    print(f't2:{t2},type:{t2.type()}')#t2中的内容没有被修改
    print("-"*40)
#3.定义函数：演示：从标量张量(只有一个值的张量)中提取其内容
def dm03():
    #3.1 创建张量
    #t1=torch.tensor(100)
    t1=torch.tensor([100])#只能有一个元素，两个会报错，例如：t1=torch.tensor([100,200])
    print(f't1:{t1},type:{t1.type()}')
    #3.2 从标量张量中提取其内容
    a=t1.item()
    print(f'value:{a},type:{type(a)}')


#测试主函数
if __name__=="__main__":
    dm01()
    dm02()
    dm03()

