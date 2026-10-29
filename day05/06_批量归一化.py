"""
案例：
    代码演示归一化，它（批量归一化）也属于正则化的一种，也是用于缓解过拟合情况的

批量归一化：
    思路：
        先对数据做标准化（会丢失一些信息），然后对数据做放缩（权重）和平移9偏置），再找补一些信息
"""
import torch
import torch.nn as nn

#1.定义函数：处理二维数据
def dm01():
    #创建图像样本数据
    #1张图片，2个通道，3行4列（像素点）
    input_2d=torch.randn(size=(1,2,3,4))
    print(f'input_2d:{input_2d}')

    #2.创建批量数据化层（BN层
    #参数一：输入特征值=图片的通道数
    #参数二：噪声值（小常数）默认1e-5
    #参数三：动量值
    #参数四：是否使用可训练的缩放和偏置参数
    bn2d=nn.BatchNorm2d(num_features=2,eps=1e-5,momentum=0.1,affine=True)

    #3.对数据进行批量归一化处理
    output_2d=bn2d(input_2d)
    print(f'output_2d:{output_2d}')

#定义函数：处理二维数据
def dm02():
    #1.创建样本
    #2行2列
    input_1d=torch.randn(size=(2,2))
    print(f'input_1d:{input_1d}')
    #2.创建批量数据化层（BN层）
    linear1=nn.Linear(in_features=2,out_features=4)
    #3.对数据进行线性变换
    l1=linear1(input_1d)
    print(f'output_1d:{l1}')
    #4.创建批量归一化层
    bn1d=nn.BatchNorm1d(num_features=4)
    #5.对线性处理的结果进行批量归一化处理
    output_1d=bn1d(l1)

    print(f'output_1d:{output_1d}')


#训练
if __name__=='__main__':
    dm01()
    dm02()