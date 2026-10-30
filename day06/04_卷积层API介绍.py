"""
案例：
    演示卷积层API的使用，用于提取图像的局部特征，获取：特征图（Feature Map)

卷积神经网络介绍：
    概况：
        全称叫做：Convolutional Neural Network，简称：CNN ，即包含卷积层的神经网络
    组成：
        卷积层（Convolutional Layer）：
            用于提取图像的局部特征，结合卷积核（每个卷积核=1个神经元）实现，处理后的结果叫：特征图（Feature Map）
        池化层（Pooling Layer）：
            用于降低特征图的维度，降低采样
        全连接层（Fully Connected Layer）：
            用于预测结果以及输出结果
    特征图的计算方式：
        N=(W-F+2P)/S+1
        W：输入图像的宽度（或高度，因为是平方所以宽度和高度一样）
        F：卷积核的宽度
        P：零填充的宽度
        S：步长的宽度
        N：输出特征图的宽度（或高度，因为是平方所以宽度和高度一样）
"""
#导包
import torch
import matplotlib.pyplot as plt
import torch.nn as nn

#1.定义函数，用于完成图像的加载，卷积和特征图可视化
def dm01():
    #1.1加载RGB真彩图
    img=plt.imread('./data/img.jpg')
    # #1.2打印读取到的图片信息
    # print(f'img:{img},shape:{img.shape}')#HWC(640,640,3)
    #1.3把图片的形状转换成模型需要的形状HWC--CHW img 封装成张量--转换维度
    img2=torch.tensor(img,dtype=torch.float32)
    img2=img2.permute(2,0,1)
    # print(f'img2:{img2},shape:{img2.shape}')
    #1.4因为这里只有一张图，所以增加一个维度，从CHW---(1,C,H,W)也就是一张3通道的640*640像素图
    img3=img2.unsqueeze(0)
    # print(f'img3:{img3},shape:{img3.shape}')
    #1.5卷积层对象，提取特征图
    #参数一：输入的通道数，参数二：输出的通道数，参数三：卷积核的大小，参数四：步长，参数五：零填充的大小
    conv=nn.Conv2d(3,4,3,2,0)
    #1.6具体的卷积计算
    conv_img=conv(img3)
    #1.7打印卷积后的结果
    print(f'conv_img:{conv_img},shape:{conv_img.shape}')#1，4，319，319
    #1.8查看提取到的四个特征图
    img4=conv_img[0]
    print(f'img4:{img4},shape:{img4.shape}')
    #1.9把上述的图像从CHW--HWC
    img5=img4.permute(1,2,0)
    #1.10可视化第1个通道
    feature1=img5[:,:,0].detach().numpy()
    plt.imshow(feature1)
    plt.show()

#3 测试
if __name__=='__main__':
    dm01()