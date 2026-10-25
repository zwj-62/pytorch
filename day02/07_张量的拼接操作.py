"""
案例：
    演示张量的拼接操作
涉及到的API:
    torch.cat()：沿着指定维度进行拼接，不改变维度，拼接张量，除了拼接的那个维度外，其他维度必须保持一致
    torch.stack()：沿着新维度进行拼接，会改变维度，拼接张量，所有的维度都必须保持一致

"""

#导包
import torch

#创建随机种子
torch.manual_seed(24)

#1.创建两个张量
t1=torch.randint(0,10,(2,3))
print(f't1:{t1},shape:{t1.shape}')
t2=torch.randint(0,10,(2,3))
print(f't2:{t2},shape:{t2.shape}')

#2.演示张量的拼接
#2.1cat()拼接两个张量
t3=torch.cat([t1,t2],dim=0)#也就是拼接行，行可以不一样，列要一样
print(f't3:{t3},shape:{t3.shape}')
t4=torch.cat([t1,t2],dim=1)#也就是拼接列，列可以不一样，行要一样
print(f't4:{t4},shape:{t4.shape}')

#2.2stack()拼接两个张量，可以是
t5=torch.stack([t1,t2],dim=0)
print(f't5:{t5},shape:{t5.shape}')
t6=torch.stack([t1,t2],dim=1)
print(f't6:{t6},shape:{t6.shape}')#两个的得到的结果不一样
t7=torch.stack([t1,t2],dim=2)
print(f't7:{t7},shape:{t7.shape}')



