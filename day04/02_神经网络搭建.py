"""
案例：
    演示神经网络的搭建流程

深度学习案例的4个步骤：
    1.准备数据
    2.搭建神经网络
    3.模拟训练
    4.模拟测试

神经网络搭建流程：
    1.定义一个类，继承nn.Module
    2.在_init_()方法中，搭建神经网络
    3.在forward（）方法中，定义数据的前向传播
"""
#导包
import torch
import torch.nn as nn
from torchsummary import summary#用于计算模型参数，查看模型结构
#todo: 1.搭建神经网络 即自定义继承nn.Module的类
class ModelDemo(nn.Module):
    #todo:在init魔法方法中，完成初始化：父类成员和神经网络的搭建
    def __init__(self):
        # 1.1初始化父类成员
        super().__init__()
        # 1.2搭建神经网络：隐藏层+输出层
        # 隐藏层1：输入特征数3，输出特征数3
        self.linear1 = nn.Linear(3, 3)
        # 隐藏层2：输入特征数3，输出特征数2
        self.linear2 = nn.Linear(3, 2)
        # 隐藏层2：输入特征数3，输出特征数2
        self.linear3 = nn.Linear(2, 2)
        # 1.3对隐藏层进行参数初始化
        # 隐藏层1
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        # 隐藏层2
        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)
    #todo:1.2前向传播：输入层--->隐藏层--->输出层
    def forward(self, x):
        #1.1 第一层隐藏层设计，加权求和和激活函数(sigmoid)
        # #分解写法
        # x=self.linear1(x)
        # x=torch.sigmoid(x)
        #合并写法
        x=torch.sigmoid(self.linear1(x))
        #1.2第二层隐藏层设计，加权求和和激活函数（relu)
        x=torch.relu(self.linear2(x))
        #1.3输出层设计：加权求和和激活函数（softmax)
        #dim=-1表示按行进行计算，一条样本一条样本的处理
        x=torch.softmax(self.linear3(x),dim=-1)
        #1.4返回预测值
        return x

#todo:2.模型训练
def train():
    #1.创建模型对象
    model=ModelDemo().to("cuda")
    #2.创建数据集样本，随机生成
    data=torch.randn(size=(5,3),device="cuda")
    print(f'data:{data}')
    print(f'data.shape:{data.shape}')
    print(f'data.requires_grad:{data.requires_grad}')
    #3.调用神经网络--->进行模型训练
    output=model(data)#这里会自动调用forward方法
    print(f'output:{output}')
    print(f'output.shape:{output.shape}')
    print(f'output.requires_grad:{output.requires_grad}')
    print('-'*40)
    #4.计算和查看模型参数
    print("==================计算模型参数======================")
    #参数一：模型对象，参数二：输入数据维度
    summary(model,input_size=(5,3),device="cuda")
    print("==================查看模型数据=======================")
    for name,param in model.named_parameters():
        print(f'name:{name},param:{param}')

#todo:3:测试
if __name__=='__main__':
    train()