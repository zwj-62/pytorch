"""
案例：
    演示张量的形状操作
涉及到的API：
    torch.reshape()：在不改变内容的前提下，对其形状进行改变
    torch.unsqueeze()：在指定位置增加一个维度
    torch.squeeze()：删除指定位置的维度
    torch.transpose()：一次只能交换两个维度
    torch.permute()：一次可以交换多个维度

    torch.view()：只能修改连续张量的形状，连续张量=内存中存储顺序和张量中显示的顺序相同
    torch.contiguous()：把不连续的张量变成连续的张量，即：基于张量中显示的顺序，修改内存中的存储顺序
    torch.is_contiguous()：判断张量是否是连续的，即：内存中存储顺序和张量中显示的顺序是否相同

需要掌握的函数：
    reshape(),unsqueeze(),permute(),view()
"""
#导包
import torch
#指定随机种子
torch.manual_seed(24)

#1.定义函数：，演示：reshape()函数
def dm01():
    #1.1定义2行3列的张量
    t1=torch.randint(1,10,size=(2,3))
    print(f't1:{t1},shape:{t1.shape},row:{t1.shape[0]},col:{t1.shape[1]}')
    #1.2通过reshape 改变张量的形状，比如3行2列，1行6列
    t2=t1.reshape(3,2)
    print(f't2:{t2},shape:{t2.shape},row:{t2.shape[0]},col:{t2.shape[1]}')
    t3=t1.reshape(1,6)
    print(f't3:{t3},shape:{t3.shape},row:{t3.shape[0]},col:{t3.shape[1]}')
    print('-'*40)
    # #1.3尝试通过reshape()函数把t1转换成2行5列
    # t4=t1.reshape(2,5)#会报错，因为前后元素不一致
    # print(f't4:{t4},shape:{t4.shape},row:{t4.shape[0]},col:{t4.shape[1]}')
#2.定义函数：演示：unsqueeze()函数和squeeze()函数
def dm02():
    #2.1定义2行3列的张量
    t1=torch.randint(1,10,size=(2,3))
    print(f't1:{t1},shape:{t1.shape}')
    #2.2在0维上添加一个维度
    t2=t1.unsqueeze(0)#(1,2,3)
    print(f't2:{t2},shape:{t2.shape}')
    #2.3在1维上添加一个维度
    t3=t1.unsqueeze(1)#(2,1,3)
    print(f't3:{t3},shape:{t3.shape}')
    #2.4在2维上添加一个维度
    t4=t1.unsqueeze(2)#(2,3,1)
    print(f't4:{t4},shape:{t4.shape}')
    # #2.5在3维上添加一个维度
    # t5=t1.unsqueeze(3)#会报错
    # print(f't5:{t5},shape:{t5.shape}')
    #2.6删除所有为1的维度
    t6=torch.randint(1,10,size=(2,1,3,1,1))
    print(f't6={t6},shape:{t6.shape}')
    t7=t6.squeeze()
    print(f't7:{t7},shape:{t7.shape}')
#3.定义函数：演示：transpose()函数和permute()函数
def dm03():
    #3.1定义一个2行3列的张量
    t1=torch.randint(1,10,size=(2,3,4))
    print(f't1:{t1},shape:{t1.shape}')
    #3.2改变维度从（2,3,4）---->(4,3,2)
    t2=t1.transpose(0,2)
    print(f't2:{t2},shape:{t2.shape}')
    #3.3改变维度从（2，3，4）----->(4,2,3)
    t3=t1.permute(2,0,1)
    print(f't3:{t3},shape:{t3.shape}')

#4.定义函数：演示：view()函数 contiguous()函数 is_contiguous()函数
def dm04():
    #思路是：演示view（）无法改变不连续的张量的形状，可以通过is_contiguous()函数来判断是否连续，也可以通过contiguous()函数来改变不连续的张量的形状
    #1.定义张量
    t1=torch.randint(1,10,size=(2,3))
    print(f't1:{t1},shape:{t1.shape}')
    #2.判断张张量是否连续
    print(f't1是否连续:{t1.is_contiguous()}')
    #3.通过view函数修改上述张量的形状从（2，3）到（3，2）
    t3=t1.view(3,2)
    print(f't3:{t3},shape:{t3.shape}')
    print(f't3是否连续:{t3.is_contiguous()}')
    #4。通过transpose()交换维度，交换了之后就不连续了
    t4=t1.transpose(0,1)
    print(f't4:{t4},shape:{t4.shape}')
    print(f't4是否连续:{t4.is_contiguous()}')
    #5.尝试把t3张量从（3，2）通过view（）转成（2，3）
    t5=t4.view(2,3)# 会报错，因为t4是不连续的
    print(f't5:{t5},shape:{t5.shape}')
    print(f't5是否连续:{t5.is_contiguous()}')
    #6.可以通过contiguous()函数来改变不连续的张量的形状
    t6=t4.contiguous().view(2,3)
    print(f't6:{t6},shape:{t6.shape}')
    print(f't6是否连续:{t6.is_contiguous()}')
#测试
if __name__ == '__main__':
    dm01()
    dm02()
    dm03()
    dm04()