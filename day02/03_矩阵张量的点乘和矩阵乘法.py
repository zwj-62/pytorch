"""
案例：
    演示矩阵张量的点乘和矩阵乘
点乘：
    要求两个张量的维度保持一致，即对应位置的元素进行乘法运算
    API:
    t1*t2
    t1.mul(t2) multiply:乘法

矩阵乘法：
    要求两个张量的维度满足矩阵乘法的条件，即第一个张量的列数等于第二个张量的行数（A行=B列）
    结果：A行B列
    API:
    t1@t2
    t1.matul(t2)
    t1.dot(t2) 扩展：只针对于一维张量有效
"""

#导包
import torch
#1.定义函数，演示张量：点乘
def dm01():
   #1.1定义第一个张量：两行三列
    t1=torch.tensor([[1,2,3],[4,5,6]])
    print(f't1:{t1}')
   #1.2定义第二个张量：两行三列
    t2=torch.tensor([[1,2,3],[4,5,6]])
    print(f't2:{t2}')
   #1.3进行张量的点乘
    t3=t1*t2
   #t3=t1.mul(t2)得到的结果一样
    print(f't3:{t3}')
    print('-'*40)
#2.定义函数，演示张量：矩阵乘法
def dm02():
    # 1.1定义第一个张量：两行三列
    t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(f't1:{t1}')
    # 1.2定义第二个张量：三行两列
    t2 = torch.tensor([[1, 2], [3, 4], [5, 6]])
    print(f't2:{t2}')
    # 1.3进行张量的矩阵乘法
    t3 = t1 @t2
    # t3=t1.matmul(t2)得到的结果一样
    print(f't3:{t3}')
    print('-'*40)
    #dot()只对一维数据有效
    t4=torch.tensor([1,2,3])
    t5=torch.tensor([1,2,3])
    t6=t4.dot(t5)
    print(f't6:{t6}')

#3.测试：
if __name__ == '__main__':
    dm01()
    dm02()

