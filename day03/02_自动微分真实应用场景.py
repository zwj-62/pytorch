"""
案例：
    演示自动微分在真实场景中的应用
结论：
    1. 前向传播计算出预测值
    2. 基于损失函数，结合预测值和真实值，计算梯度
    3.结合梯度更新公式：w_new = w_old - learning_rate * gradient，更新权重

"""
#导包
import torch
#定义x,表示：特征（（输入数据），假设：2行5列全1矩阵
x = torch.ones(2, 5)
print(f'x={x}')
#定义y,表示：标签（输出数据），假设：2行3列的全0矩阵
y = torch.zeros(2, 3)
print(f'y={y}')
#3.初始化（可自动微分）权重和偏置值
w = torch.randn(5, 3, requires_grad=True)
print(f'w={w}')

b = torch.randn(3, requires_grad=True)
print(f'b={b}')
#4.前向传播，计算预测值
z=torch.matmul(x, w) + b
print(f'z={z}')
#5.定义损失函数
criterion = torch.nn.MSELoss()
loss = criterion(z, y)
print(f'loss={loss}')
#6.反向传播，计算梯度
loss.backward()
#7.打印w b 用来更新的梯度
print(f'w.grad={w.grad}')
print(f'b.grad={b.grad}')



