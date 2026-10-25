"""
案例：
    演示张量的常用运算函数
涉及到的API（）如下：
    sum() 求和 mean() 平均值 max() 最大值 min() 最小值  ----->都有dim参数 0表示列 1表示行
    pow() 幂  sqrt() 开平方  exp() 自然对数  log() 自然对数  log10() 以10为底的对数----->都有dim参数
需要掌握的函数：
    sum() max() min() mean()


"""
#导包
import torch

#1.定义张量，记录初值
t1=torch.tensor(
    [[1,2,3],
     [4,5,6]]
    ,dtype=torch.float32
)
print(f't1:{t1}')
#2.演示 有dim参数的函数
#2.1求和sum()
print(t1.sum(dim=0))#列求和
print(t1.sum(dim=1))#行求和
print(t1.sum())#整体求和
print('-'*40)

#2.2max()求最大值，min()同理
print(t1.max(dim=0))#列最大值
print(t1.max(dim=1))#行最大值
print(t1.max())#整体最大值
print('-'*40)

#2.3mean()求平均值 要求张量的元素是float类型
print(t1.mean(dim=0))#列平均值
print(t1.mean(dim=1))#行平均值
print(t1.mean())#整体平均值
print('-'*40)

#3演示 没有dim值的函数
#3.1pow()求幂
print(t1.pow(2))
print(t1.pow(3))
print(t1**3)
print('-'*40)

#3.2sqrt()开平方
print(t1.sqrt())
print('-'*40)
#3.3exp()自然对数 e 的n 次幂
print(t1.exp())
print('-'*40)
#3.4log()自然对数
print(t1.log())
print('-'*40)
#3.5log10()以10为底的对数
print(t1.log10())
print('-'*40)




