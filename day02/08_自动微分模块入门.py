"""
案例：
    演示自动微分，具体如何求导
回顾：
    权重更新公式：
    w新=w旧-学习率*梯度
    梯度：损失函数的导数
    关于损失函数的导数我们无需进行计算，因为非常常用，所有pytorch模块内置里有自动微分模块，专门针对于不同的损失函数求导
    从而实现 结合 反向传输 更新 权重参数w和偏执参数b

"""
#导包
import torch

#1.定义变量，记录：出事的权重w(旧)
#参1：初始值，参2：是否自动微分，参3：数据类型
w=torch.tensor(10,requires_grad=True,dtype=torch.float32)

#2.定义loss变量，表示损失函数
loss=2*w**2 #loss求导后为4w

#3.打印梯度函数类型（用于了解）
print(f'loss的梯度函数类型：{loss.grad_fn}')
print(loss.sum())

#4.计算梯度，梯度=损失函数对权重的导数，计算完毕之后会记录到w.grad属性中
loss.backward()
print(f'w的梯度：{w.grad}')

#5.带入权重更新公式，更新权重w w新=w旧-学习率*梯度
w.data=w.data-0.01*w.grad

#6.打印更新后的权重w
print(f'更新后的权重w:{w}')


