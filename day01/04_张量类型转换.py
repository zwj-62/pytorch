"""
案例
    创建指定类型的张量：

涉及到的函数：
    type(torch 支持的数据类型)
    half() double() float() int() long()
你要掌握的函数：
    type()
"""
#导包
import torch
#场景一：直接创建指定类型的张量
t1=torch.tensor([1,2,3,5,7,8],dtype=torch.float)#默认类型float32
print(f't1:{t1},(元素)类型：{t1.dtype},t1张量的类型：{type(t1)}')
print('-'*40)
#场景二：创建好张量后，再转换类型
#思路一：type()函数，推荐使用
t2=t1.type(torch.int16)
print(f't2:{t2},(元素)类型：{t2.dtype},t2张量的类型：{type(t2)}')
print('-'*40)
#思路二：half(),double(),float(),int(),long()
print(t2.half())#half是float16，即float16类型
print(t2.double())#double是float64，即float64类型
print(t2.float())#float是float32，即float32类型，默认类型
print(t2.int())#int是int32，即int32类型
print(t2.long())#long是int64，即int64类型，默认类型
print('-'*40)

