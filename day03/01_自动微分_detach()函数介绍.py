"""
案例“
    演示detach（）的功能，解决自动微分的弊端
回顾：
    自动微分=求导”:即基于损失函数计算梯度
    结合权重更新公式：w新=w旧-学习率*梯度，来更新权重
问题：
    一个张量一旦设置了自动微分，那这个张量就不能直接转换为numpy的ndarray对象了，需要通过detach()函数来解决

"""
#导包
import torch
import numpy as np
#1：定义张量，
#参1：初始值，参2：是否自动微分，参3：数据类型
t1=torch.tensor([10,20],requires_grad=True,dtype=torch.float32)
print(f't1={t1},type:{t1.type()}')
# #2尝试把上面t1这个张量转为numpy的ndarray对象---->不行，因为t1是自动微分的，不能直接转换
# t2=t1.numpy()
# print(f't2={t2},type:{type(t2)}')
#3；解决办法，可以通过detach()来解决
t3=t1.detach()
print(f't3={t3},type:{type(t3)}')
#4.测试t1与t3是否共享内存---->内存共享
t1.data[0]=100
print(f't1={t1},type:{t1.type()}')
print(f't3={t3},type:{type(t3)}')
#5.查看t1 t3是否会自动微分
print(f't1.requires_grad={t1.requires_grad}')
print(f't3.requires_grad={t3.requires_grad}')
#6.把t3转换为numpy的ndarray对象
t4=t3.numpy()
print(f't4={t4},type:{type(t4)}')
#7.最终版: 可以直接采用链式法则，即先detach()再转换为numpy的ndarray对象
t5=t1.detach().numpy()
print(f't5={t5},type:{type(t5)}')






