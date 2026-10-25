"""
案例：
    涉及到的API：
    torch.add()  torch.sub()   torch.mul()   torch.div() torch.neg() ---->加减乘除，取反substract multiply,divide
    torch.add_()  torch.sub_()   torch.mul_()   torch.div_() torch.neg_() ---->功能同上，只不过可以修改原来的数据，类似于Pandas部分的inplace=True

需要记忆的部分：
    1：可以用+ - * /直接代替上述的加减乘除函数
    2：如果是张量和数值进行运算，那么该数值会和张量中的每个值依次进行对应的运算
"""
#导包
import torch

#1：创建张量
t1=torch.tensor([1,2,3])

#2:进行运算演示
#2.1演示加法：
t2=t1.add(10) #在t1的基础上加10，但是t1不会改变，等同于 t2=t1+10
# t2=t1.add_(10) #在t1的基础上加10，但是t1会改变,等同于 t1=t1+10 t2=t1
#其他的函数效果同上，这里就快速演示了：
# t2=t1.sub(1) #在t1的基础上减1，但是t1不会改变，等同于 t2=t1-1
# t2=t1-1 在t1的基础上减1，但是t1不会改变，等同于 t2=t1-1
# t2=t1.mul(2) #在t1的基础上乘2，但是t1不会改变，等同于 t2=t1*2
# t2=t1*2 在t1的基础上乘2，但是t1不会改变，等同于 t2=t1*2
# t2=t1.div(2) #在t1的基础上除2，但是t1不会改变，等同于 t2=t1/2
# t2=t1/2 在t1的基础上除2，但是t1不会改变，等同于 t2=t1/2
# t2=t1.neg()


#测试
print(f't1:{t1}')
print(f't2:{t2}')