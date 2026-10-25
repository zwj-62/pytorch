"""
案例：
    绘制激活Tanh函数图像
tanh 激活函数介绍：
    激活函数的目的：
    给模型增加非线性功能，让模型（神经玩）既可以做分类，还可以做线性回归
激活函数分类：
    tanh
    Tanh
    ReLU
    Softmax
tanh 激活函数：
    主要应用于二分类的输出层，且适用于 浅层神经网络（不超过5层）
    数据在[-6,6]之间有效果，在此范围外效果很差，在[-3,3]之间效果最明显
    求导后范围在[0,0.25]之间
    tanh函数公式：1/(1+e^-x)
Tanh:
    主要应用于隐藏层，且适用于浅层神经网络（不超过5层）
    数据在[-3，3]之间有效果，在此范围外效果很差，在[-1,1]之间效果最明显
    求导后范围在[0,1]之间
    Tanh函数公式：(e^x-e^-x)/(e^x+e^-x)
细节：
    绘制激活函数图像时出现以下提示：需要将anaconda3等

"""
#导包
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
#1.创建画布和坐标轴，1行2列
fig, axes = plt.subplots(1, 2)
#2.生成-20-20之间的1000个数据点
x=torch.linspace(-20,20,steps=1000)
# print(f'x={x}')
#3.计算上述1000个点，tanh激活函数处理后的数
y=torch.tanh(x)
# print(f'y={y}')
#4.绘制图像
axes[0].plot(x,y)
axes[0].set_title('tanh激活函数图像')
axes[0].grid()
#5.在第二个图上，绘制tanh激活函数导数图像
#5.1重新生成-20-20之间的1000个数据点
#参数一：起始值，参数二：结束值，参数三：数据点个数,参数四：是否需要求导
x=torch.linspace(-20,20,steps=1000,requires_grad=True)
#5.2具体的计算以上1000个点在tanh激活函数导数处理后的结果
torch.tanh(x).sum().backward()
#5.3绘制图像
axes[1].plot(x.detach(),x.grad)
axes[1].set_title('tanh激活函数导数图像')
axes[1].grid()
plt.show()




