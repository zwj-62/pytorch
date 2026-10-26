"""
案例：
    演示参数初始化的7种方法

参数初始化的目的：
    1.防止梯度消失，或者梯度爆炸
    2.提高收敛的速度
    3.打破对称性

参数初始化的方法：
    无法打破对称性的：
        全0 全1 固定值
    可以打破对称性的：
        随机初始化，正态分布初始化，kaiming初始化，xavier初始化

总结：
    1.记忆kaiming初始化，xavier初始化和全0初始化
    2.关于初始化的选择上：
        激活函数ReLU及其系列激活函数，推荐使用kaiming初始化
        激活函数非ReLU，推荐使用xavier初始化
        如果是浅层网络，可以考虑随机初始化

"""
#导包
import torch.nn as nn #neutral network 神经网络
#1.均匀分布随机初始化:
def dm01():
    #1.1创建一个线形层，输入维度5，输出维度3
    linear = nn.Linear(5,3)
    #1.2对权重（weight)进行随机初始化，从0-1均匀分布产生参数
    nn.init.uniform_(linear.weight)
    #1.3对偏置（bias）进行随机初始化，从0-1均匀分布产生参数
    nn.init.uniform_(linear.bias)
    #1.4打印生成的结果
    print(linear.weight.data)
    print(linear.bias.data)
#2.固定初始化
def dm02():
    #1.1创建一个线形层，输入维度5，输出维度3
    linear = nn.Linear(5,3)
    #1.2对权重（weight)进行固定值初始化，固定值为3
    nn.init.constant_(linear.weight, 3)
    #1.3对偏置（bias）进行固定值初始化，固定值为3
    nn.init.constant_(linear.bias, 3  )
    #1.4打印生成的结果
    print(linear.weight.data)
    print(linear.bias.data)

#3.全0初始化
def dm03():
    #1.1创建一个线形层，输入维度5，输出维度3
    linear = nn.Linear(5,3)
    #1.2对权重（weight)进行全0初始化，全0初始化
    nn.init.zeros_(linear.weight)
    #1.3对偏置（bias）进行全0初始化，全0初始化
    nn.init.zeros_(linear.bias)
    #1.4打印生成的结果
    print(linear.weight.data)
    print(linear.bias.data)
#4.全1初始化
def dm04():
    #1.1创建一个线形层，输入维度5，输出维度3
    linear = nn.Linear(5,3)
    #1.2对权重（weight)进行全1初始化，全1初始化
    nn.init.ones_(linear.weight)
    #1.3对偏置（bias）进行全1初始化，全1初始化
    nn.init.ones_(linear.bias)
    #1.4打印生成的结果
    print(linear.weight.data)
    print(linear.bias.data)
#5.正态分布随机初始化
def dm05():
    #1.1创建一个线形层，输入维度5，输出维度3
    linear = nn.Linear(5,3)
    #1.2对权重（weight)进行正态分布初始化，正态分布初始化
    nn.init.normal_(linear.weight)
    #1.3对偏置（bias）进行正态分布初始化，正态分布初始化
    nn.init.normal_(linear.bias)
    #1.4打印生成的结果
    print(linear.weight.data)
    print(linear.bias.data)
#6.kaiming初始化
def dm06():
    #1.1创建一个线形层，输入维度5，输出维度3
    linear = nn.Linear(5,3)
    #1.2对权重（weight)进行kaiming正态分布初始化，kaiming正态分布初始化
    nn.init.kaiming_normal_(linear.weight)
    # #1.2对权重（weight)进行kaiming均匀分布初始化，kaiming均匀分布初始化
    # nn.init.kaiming_uniform_(linear.weight)
    #1.4打印生成的结果
    print(linear.weight.data)

#7.xacier 初始化
def dm07():
    #分为正态分布和均匀分布两种初始化
    #1.1创建一个线形层，输入维度5，输出维度3
    linear = nn.Linear(5,3)
    #1.2对权重（weight)进行xavier正态分布初始化，xavier正态分布初始化
    nn.init.xavier_normal_(linear.weight)
    # #1.2对权重（weight)进行xavier均匀分布初始化，xavier均匀分布初始化
    # nn.init.xavier_uniform_(linear.weight))
    #1.4打印生成的结果
    print(linear.weight.data)

if __name__=='__main__':
    dm01()
    dm02()
    dm03()
    dm04()
    dm05()
    dm06()
    dm07()