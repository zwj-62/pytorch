"""
案例：
    梯度下降优化方法案例
梯度下降相关介绍：
    概述：
        梯度下降是一种结合本次损失函数的导数（作为梯度）基于学习率来更新权重的一种方法。
    公式：
        W新=W旧-学习率*梯度（本次）
    存在的问题：
        1。遇到平缓区域时，梯度下降（权重更新）可能很慢
        2.可能遇到鞍点（也就是梯度0 的情况）
        3.可能遇到局部最小值
    解决思路：
        从上述的学习率或者梯度入手，进行优化。于是有了：动量法Momentum、自适应学习法：Adagrad、RMSprop、综合衡量：Adam

    动量法Momentum：
        公式：
            St=beta*(St-1)+(1-beta)*Gt
        解析：
            St:本次的指数移动加权平均的结果
            beta:调节权重系数，越大数据越平缓，历史指数移动加权平均的比重越大，本次梯度的权重越小
            St-1:上一次的指数移动加权平均的结果
            Gt:本次计算的梯度（不考虑历史梯度—）
        加入动量法Momentum之后的梯度更新公式：
            W新=W旧-学习率*St

    自适应学习法Adagrad：
            公式：
                累计平方梯度：
                     St=(St-1)+Gt*Gt
                解析：
                    St:本次的指数移动加权平均的结果
                    St-1:上一次的指数移动加权平均的结果
                    Gt:本次计算的梯度（不考虑历史梯度—）
                学习率：
                    学习率=学习率/(1+sqrt(St)+小常数)
                    解释：
                        小常数：1e-10,目的：防止分母变为0
            梯度下降公式：
                W新=W旧-调整后的学习率*（本次）梯度Gt
            缺点：
                可能会导致学习率过早过量的降低，导致模型后期学习率太小，难以找到最优解

        自适应学习率：RMSprop：可以看着是Adagrad的升级版入了调和系数
            公式：
                累计平方梯度：
                     St=beta(St-1)+(1-beta)*Gt*Gt
                解析：
                    St:本次的指数移动加权平均的结果
                    St-1:上一次的指数移动加权平均的结果
                    Gt:本次计算的梯度（不考虑历史梯度—）
                学习率：
                    学习率=学习率/(1+sqrt(St)+小常数)
                    解释：
                        小常数：1e-10,目的：防止分母变为0
            梯度下降公式：
                W新=W旧-调整后的学习率*（本次）梯度Gt
            优点：
                通过引入调和系数，控制历史梯度对历史信息获取的多少
        自适应矩估计：Adam
            思路：即优化学习率，又优化梯度
            公式：
                一阶矩：
                    Mt=beta(Mt-1)+(1-beta)*Gt 充当梯度
                    St=beta(St-1)+(1-beta)*Gt*Gt 充当学习率
                二阶矩：
                    Mt^=(Mt)/(1-beta^t) 充当梯度
                    St^=(St)/(1-beta^t) 充当学习率
                梯度下降公式：
                    W新=W旧-学习率/(squrt(S^)+小常数)*Mt^
            大白话翻译：
                Adam=RMSProp+Momentum
需要掌握：
    1.动量法Momentum
    2.自适应矩估计：Adam
"""
#导包
import torch
import torch.nn as nn
#1.定义函数：动量法Momentum
def dm01():
    #1.1初始化权重参数
    w=torch.tensor(1.0,requires_grad=True,dtype=torch.float32)
    #1.2定义损失函数：
    criterion=((w**2)/2.0)
    #1.3创建优化器（函数对象）基于SGD（随机梯度下降），加入参数momentum就是动量法Momentum
    optimizer=torch.optim.SGD([w],lr=0.01,momentum=0.9)
    #1.4计算梯度值：梯度清零+反向传播+参数更新
    optimizer.zero_grad()
    criterion.backward()
    optimizer.step()
    print(f"w:{w},w.grad:{w.grad}")
    #1.5重复上述步骤，第二次更新权重
    #1.5.1定义损失函数‘
    criterion2=((w**2)/2.0)
    #1.5.2计算梯度值：梯度清零+反向传播+参数更新
    optimizer.zero_grad()
    criterion2.backward()
    optimizer.step()
    print(f"w:{w},w.grad:{w.grad}")
#2定义函数：Adagrad
def dm02():
    #1.1初始化权重参数
    w=torch.tensor(1.0,requires_grad=True,dtype=torch.float32)
    #1.2定义损失函数：
    criterion=((w**2)/2.0)
    #1.3创建优化器（函数对象）
    # 思路一：基于SGD（随机梯度下降），加入参数momentum就是动量法Momentum
    #参数一：待优化的参数，参数二：学习率，参数三：动量系数
    optimizer=torch.optim.SGD([w],lr=0.01,momentum=0.9)
    #思路二：基于Adagrad的优化器
    optimizer=torch.optim.Adagrad([w],lr=0.01)
    #1.4计算梯度值：梯度清零+反向传播+参数更新
    optimizer.zero_grad()
    criterion.backward()
    optimizer.step()
    print(f"w:{w},w.grad:{w.grad}")
    #1.5重复上述步骤，第二次更新权重
    #1.5.1定义损失函数‘
    criterion2=((w**2)/2.0)
    #1.5.2计算梯度值：梯度清零+反向传播+参数更新
    optimizer.zero_grad()
    criterion2.backward()
    optimizer.step()
    print(f"w:{w},w.grad:{w.grad}")
#3定义函数：RMSprop
def dm03():
    #1.1初始化权重参数
    w=torch.tensor(1.0,requires_grad=True,dtype=torch.float32)
    #1.2定义损失函数：
    criterion=((w**2)/2.0)
    #1.3创建优化器（函数对象）
    # 思路一：基于SGD（随机梯度下降），加入参数momentum就是动量法Momentum
    #参数一：待优化的参数，参数二：学习率，参数三：动量系数
    optimizer=torch.optim.SGD([w],lr=0.01,momentum=0.9)
    # #思路二：基于Adagrad的优化器
    # optimizer=torch.optim.Adagrad([w],lr=0.01)
    #思路三：基于RMSprop的优化器
    optimizer=torch.optim.RMSprop([w],lr=0.01,alpha=0.99)
    optimizer.zero_grad()
    criterion.backward()
    optimizer.step()
    print(f"w:{w},w.grad:{w.grad}")
    #1.5重复上述步骤，第二次更新权重
    #1.5.1定义损失函数‘
    criterion2=((w**2)/2.0)
    #1.5.2计算梯度值：梯度清零+反向传播+参数更新
    optimizer.zero_grad()
    criterion2.backward()
    optimizer.step()
    print(f"w:{w},w.grad:{w.grad}")
#4定义函数：Adam
def dm04():
    #1.1初始化权重参数
    w=torch.tensor(1.0,requires_grad=True,dtype=torch.float32)
    #1.2定义损失函数：
    criterion=((w**2)/2.0)
    #1.3创建优化器（函数对象）
    # 思路一：基于SGD（随机梯度下降），加入参数momentum就是动量法Momentum
    #参数一：待优化的参数，参数二：学习率，参数三：动量系数
    optimizer=torch.optim.SGD([w],lr=0.01,momentum=0.9)
    # #思路二：基于Adagrad的优化器
    # optimizer=torch.optim.Adagrad([w],lr=0.01)
    # #思路三：基于RMSprop的优化器
    # optimizer=torch.optim.RMSprop([w],lr=0.01,alpha=0.99)
    #思路四：基于Adam的优化器
    optimizer=torch.optim.Adam([w],lr=0.01,betas=(0.9,0.99))
    optimizer.zero_grad()
    criterion.backward()
    optimizer.step()
    print(f"w:{w},w.grad:{w.grad}")
    #1.5重复上述步骤，第二次更新权重
    #1.5.1定义损失函数‘
    criterion2=((w**2)/2.0)
    #1.5.2计算梯度值：梯度清零+反向传播+参数更新
    optimizer.zero_grad()
    criterion2.backward()
    optimizer.step()
    print(f"w:{w},w.grad:{w.grad}")
if __name__ == '__main__':
    dm01()
    dm02()
    dm03()
    dm04()