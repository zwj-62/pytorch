"""
案例：
    代码演示，随机失活

正则化的作用：
    环节模型过拟合的情况

正则化的方式：
    L1正则化：权重可以降为0，相当于降维
    L2正则化：权重可以无线接近0
    Dropout：随机失活，相当于随机删除一些神经元，防止一些特征结果对结果的影响比较大
    Batch Normalization：批量归一化，相当于对输入进行归一化处理，防止梯度消失和梯度爆炸
"""
#导包
import torch
import torch.nn as nn
#随机种子设置
torch.manual_seed(24)

#1.定义函数：演示随机失活Dropout
def dm01():
    #1创建隐藏层输出结果
    t1=torch.randint(0,10,size=(1,4)).float()
    print(f't1:{t1}')
    #2进行下一层的加权求和和激活函数的计算
    #2.1创建全连接层
    #参数一:输入特征维度，参数二：输出特征维度
    linear=nn.Linear(4,5)
    #2.2加权求和
    l1=linear(t1)
    print(f'l1:{l1}')
    #2.3激活函数
    output=torch.relu(l1)
    print(f'l1激活:{output}')
    #3随机失活
    dropout=nn.Dropout(p=0.5)
    output2=dropout(output)
    print(f'随机失活:{output2}')

#测试：
if __name__=='__main__':
    dm01()