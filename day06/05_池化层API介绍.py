"""
案例：
    演示池化层的相关操作，

池化层解释：
    目的：
        降低特征图的维度
    思路：
        最大池化
        平均池化
    特点：
        池化不会改变数据的通道数

"""
#导包
import torch
import torch.nn as nn
#1.定义函数，演示单通池化
def dm01():
    #1.1创建1个通道3*3的二维矩阵
    inputs=torch.tensor([
        [
            [0,1,2],
            [3,4,5],
            [6,7,8],
        ]
    ])
    print(f'inputs:{inputs},shape:{inputs.shape}')#1,3,3

    #1.2创建最大池化层
    # 参数一：池化核的大小；参数二：步长；参数三：填充大小
    max_pool=nn.MaxPool2d(2,1,0)
    outputs=max_pool(inputs)
    print(f'outputs:{outputs},shape:{outputs.shape}')#1,2,2
#2.定义函数，演示多通道池化
def dm02():
    #2.1创建3个通道3*3的二维矩阵
    inputs=torch.tensor([
        [
            [0,1,2],
            [3,4,5],
            [6,7,8],
        ],
        [
            [10,20,30],
            [40,50,60],
            [70,80,90],
        ],
        [
            [11,22,33],
            [44,55,66],
            [77,88,99],
        ]
    ])
    print(f'inputs:{inputs},shape:{inputs.shape}')#3,3,3
    #2.2创建最大池化层
    max_pool=nn.MaxPool2d(2,1,0)
    outputs=max_pool(inputs)
    print(f'outputs:{outputs},shape:{outputs.shape}')#3,2,2
    #2.3创建平均池化层
    avg_pool=nn.AvgPool2d(2,1,0)
    outputs=avg_pool(inputs)
    print(f'outputs:{outputs},shape:{outputs.shape}')#3,2,2

#3.测试
if __name__=='__main__':
    dm01()
    dm02()