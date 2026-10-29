"""
案例：
    ANN（人工神经网络）案例：手机价格分类案例
背景：
    基于手机的20个案例，预测手机价格区间（四个区间），可以用机器学习做也可以用深度学习做，这里用ANN来做，ANN是人工神经网络模型

ANN案例的实现步骤；
    1.构建数据集
    2.搭建神经网络
    3.模型训练
    4.模型测试
优化思路：
    1.SGD---Adam
    2.学习率---0.01---0.001---0.0001
    3.对数据进行标准化
    4.增加网络深度
    5.调整训练轮数
    6.。。。。
"""
#导包
import torch                                            #pytorch 框架，封装了张量的各种操作
from torch.utils.data import TensorDataset              #数据集对象，数据---Tensor ---数据库---数据加载器
from torch.utils.data import DataLoader                 #数据加载器
from torch import nn                                    #neural network 封装了神经网络的各种操作
from torch import optim                                 #优化器
from sklearn.model_selection import train_test_split    #训练集和测试集的划分
import matplotlib.pyplot as plt                         #绘图
import numpy as np                                      #数组/矩阵操作
import pandas as pd                                     #数据处理
import time                                             #时间模块

from torchsummary import summary


#todo 1.定义函数，构建数据集
def create_dataset():
    #1.加载csv数据D:\pytorch\day06\data\手机价格预测.csv
    data=pd.read_csv(r'D:\pytorch\day06\data\手机价格预测.csv')
    # print(f'data:{data.head()}')
    # print(f'data.shape:{data.shape}')#2000 ,21
    #2.获取x的特征列和y的标签列
    x,y=data.iloc[:,:-1],data.iloc[:,-1]
    # print(f'x.shape:{x.shape},y.shape:{y.shape}')
    #3.把特征列转换成浮点数
    x=x.astype(np.float32)
    # print(f'x:{x.head()}')
    #4.把数据集划分为训练集和测试集
    #参数一：特征，参数二：标签，参数三：测试集比例，参数四：随机种子，参数五：分层采样
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=3,stratify=y)
    #5.把数据集封装成张量数据集，思路：数据---张量Tensor---数据集TensorDataSet---数据加载器DataLoader
    train_dataset=TensorDataset(torch.tensor(x_train.values),torch.tensor(y_train.values))
    test_dataset=TensorDataset(torch.tensor(x_test.values),torch.tensor(y_test.values))
    # print(f'train_dataset:{train_dataset}')
    # print(f'test_dataset:{test_dataset}')
    #x_train_shape[1]是特征的维度20，len(np.unique(y))是标签的类别数4
    return train_dataset,test_dataset,x_train.shape[1],len(np.unique(y))

#todo 2.构建神经网络
class PhonePriceModel(nn.Module):
    def __init__(self,input_dim,output_dim):
        # 1.1在init魔法方法中，初始化父类成员，以及搭建神经网络
        super().__init__()
        #1.2搭建神经网络：隐藏层+输出层
        #隐藏层1
        self.linear1=nn.Linear(input_dim,128)
        #隐藏层2
        self.linear2=nn.Linear(128,64)
        #输出层
        self.linear3=nn.Linear(64,output_dim)
    #2定义前向传播方法，forword()
    def forward(self,x):
        #2.1隐藏层1：加权求和+激活函数（relu)
        x=torch.relu(self.linear1(x))
        #2.2隐藏层2：加权求和+激活函数（relu)
        x=torch.relu(self.linear2(x))
        #2.3输出层：加权求和+激活函数（softmax)
        # x=torch.softmax(self.linear3(x),dim=1)
        #正常写法是2.3但是不需要，因为后面使用的是交叉熵损失函数，已经集成了softmax
        return self.linear3(x)

#todo 3.模型训练
def train(train_dataset,test_dataset,input_dim,output_dim):
    #1.创建数据加载器
    #参数一：数据集，1600个 因为0.2，0.8*2000=1600个训练集，0.2*2000=400个测试集
    # 参数一：数据加载器，参数二：批次大小，参数三：是否打乱数据（训练集打乱，测试集不打乱）
    train_loader=DataLoader(train_dataset,batch_size=16,shuffle=True)
    #2.创建神经网络模型
    model=PhonePriceModel(input_dim,output_dim)
    #3.定义损失函数，因为是多分类，所以是CrossEntropyLoss，也就是多分类交叉熵损失函数
    criterion=nn.CrossEntropyLoss()
    #4.定义优化器，参数一：模型参数，参数二：学习率
    optimizer=optim.Adam(model.parameters(),lr=0.001)
    #5.模型训练
    #5.1定义变量，记录训练的轮次
    epoch=50
    #5.2开始对每轮的训练
    for i in range(epoch):
        #5.2.1定义变量，记录每轮的损失
        total_loss,batch_num=0.0,0
        #5.2.2定义变量，记录训练开始的时间
        start_time=time.time()
        #5.2.3开始对每批次数据的训练
        for x,y in train_loader:
            #5.2.4 模型切换
            model.train()#训练模式，model.eval测试模式
            #5.2.5模型预测
            y_pred=model(x)
            #5.2.6计算损失
            loss=criterion(y_pred,y)
            #5.2.7梯度清零+反向传播+优化参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #5.2.8记录每轮的损失，以及批次数，用于后续计算每轮的平均损失值
            total_loss+=loss.item()#记录每轮的损失，用于后续计算每轮的平均损失值
            batch_num+=1
        #5.4至此，一轮训练结束，打印每轮的损失值

        print(f'epoch:{i+1},loss:{total_loss/batch_num:.4f},time:{time.time()-start_time:.2f}s ')

    #6.模型保存
    #参数一：模型参数，参数二：保存路径
    print(f'\n\n模型的参数信息：{model.state_dict()}\n\n')
    torch.save(model.state_dict(),'./model/phone.pth')

#todo 4.模型测试
def evaluate(test_dataset,input_dim,output_dim):
    #1.创建神经网络分类对象
    model=PhonePriceModel(input_dim,output_dim)
    #2.加载模型参数
    model.load_state_dict(torch.load('./model/phone.pth'))
    #3.创建测试集的数据加载器对象
    test_loader=DataLoader(test_dataset,batch_size=8,shuffle=False)
    #4.定义变量，记录预测正确的样本数
    correct=0
    #5.开始对每批次数据的测试
    for x,y in test_loader:
        #5.1模型切换
        model.eval()
        #5.2模型预测
        y_pred=model(x)
        print(f'y_pred:{y_pred}')
        #5.3根据加权求和，得到分类，用argmax（）获取最大值对应的下标也就是分类
        y_pred=torch.argmax(y_pred,dim=1)
        # print(f'y_pred:{y_pred}')
        # print(f'y:{y}')
        #5.4判断预测是否正确，
        # print(y_pred==y)
        # print(y_pred==y).sum()
        correct+=(y_pred==y).sum()
    #6.至此，模型预测结束，打印测试的准确率
    print(f'test_accuracy:{correct/len(test_dataset):.4f}')
#todo 5.测试
if __name__ == '__main__':
    #1.准备数据集
    train_dataset, test_dataset, input_dim, output_dim = create_dataset()
    # print(f'训练集数据集对象：{train_dataset}')
    # print(f'测试集数据集对象：{test_dataset}')
    # print(f'输入特征维度：{input_dim},输出标签类别数： {output_dim}')
    #2.搭建神经网络模型
    model=PhonePriceModel(input_dim,output_dim)
    #计算模型参数
    #参数一：模型对象 #参数二：输入数据的形状（批次大小和输入的特征数）每批16条，每条20个特征
    summary(model, input_size=(input_dim,), batch_size=16, device='cpu')
    #3.模型训练
    train(train_dataset,test_dataset,input_dim,output_dim)
    #4模型测试
    evaluate(test_dataset,input_dim,output_dim)

