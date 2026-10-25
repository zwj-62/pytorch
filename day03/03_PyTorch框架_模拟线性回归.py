#导入相关模块：
import torch
from torch.utils.data import TensorDataset #构造数据集对象
from torch.utils.data import DataLoader #数据加载器
from torch import nn #神经网络模块。nn中有平方损失函数和假设函数
from torch import optim#optim中有优化器模块
from sklearn.datasets import make_regression#创建线性回归数据集
import matplotlib.pyplot as plt#绘图模块

plt.rcParams['font.sans-serif'] = ['SimHei']#解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False#解决负号显示问题


#1.定义函数，创建线性回归样本数据
def create_dataset():
    x,y,coef=make_regression(
        n_samples=100,#样本数量
        n_features=1,#特征数量
        noise=10,#噪声
        coef=True,#是否返回系数
        bias=14.5,#偏置
        random_state=3,#随机种子
    )
    #2.把上述的数据封装成张量对象
    #make_regression() 返回的是 NumPy 数组。这两行把它们转换成 PyTorch 张量，方便后续交给 PyTorch 模型计算。
    x=torch.tensor(x,dtype=torch.float32)
    y=torch.tensor(y,dtype=torch.float32)
    #返回结果
    return x,y,coef

#2.定义函数，表示模型训练
def train(x,y,coef):
    #创建数据对象，把tensor---->数据集对象---->数据加载器对象
    dataset=TensorDataset(x,y)
    #2.创建数据加载器对象，
    # 参数一：数据集对象，参数二：批量大小，参数三：是否打乱数据
    dataloader=DataLoader(dataset,batch_size=16,shuffle=True)
    #3.创建初始的线性回归模型
    #参数一：输入特征数量，参数二：输出特征数量
    model=nn.Linear(in_features=1,out_features=1)
    #4.创建损失函数对象
    criterion=nn.MSELoss()
    #5.创建优化器对象
    #参数一：模型参数，参数二：学习率
    optimizer=optim.SGD(model.parameters(),lr=0.01)
    #6.具体训练过程
    #6.1定义变量，分别表示：训练轮数，每轮的平均损失，训练总损失值，训练的样本数
    epochs,loss_list,total_loss,total_sample=100,[],0,0
    #6.2开始训练，按轮进行训练
    for epoch in range(epochs):#一共100轮次：0，1....99
        #6.3每轮是分批次进行计算的，所以从数据加载器中获取本批次的数据
        for train_x,train_y in dataloader:#一共是7个批次，16，16，16，16，16，16，4
            #6.4模型预测：
            y_pred=model(train_x)
            #6.5计算每批次的损失值
            loss=criterion(y_pred,train_y.reshape(-1,1))#-1自动计算
            #6.6计算总损失和样本批次数
            total_loss+=loss.item()
            total_sample+=1
            #6.7 梯度清零，反向传播，更新参数
            optimizer.zero_grad()#梯度清零
            loss.backward()#反向传播
            optimizer.step()#更新参数
        #6.8 记录每轮的平均损失值
        loss_list.append(total_loss/total_sample)
        print(f'第{epoch+1}轮的平均损失值为：{total_loss/total_sample}')
    #7.打印最终训练结果
    print(f'{epochs}轮的平均损失分别为：{loss_list}')
    print(f'模型的参数为：{model.weight},偏置为：{model.bias}')
    #8.绘制损失值变化图像
    plt.plot(range(epochs),loss_list)
    plt.title('损失值变化图像')
    plt.xlabel('轮数')
    plt.ylabel('损失值')
    plt.grid()
    plt.show()
    #9.绘制预测点和真实点图像
    #9.1绘制样本点分布情况
    plt.scatter(x,y)
    #9.2绘制训练模型的预测值
    #x:100个样本特征点
    y_pred=torch.tensor(data=[v*model.weight+model.bias for v in x])
    #9.3计算真实值
    y_true=torch.tensor(data=[v*coef+14.5 for v in x])
    #9.4绘制预测值和真实值的折线图
    plt.plot(x,y_pred,color='red',label='预测值')
    plt.plot(x,y_true,color='blue',label='真实值')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    x,y,coef=create_dataset()
    # print(f'x={x},y={y},coef={coef}')
    train(x,y,coef)

