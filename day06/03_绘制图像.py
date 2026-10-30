"""
案例：
    演示基础的图像操作

图像的分类：
    二值图：1通道，每个像素点是由0，1组成
    灰度图：1通道，每个像素点是由0-255组成
    索引图：1通道，每个像素点是由0-255组成，但每个像素点的值表示的是索引，索引对应的值在palettes中
    RGB图：3通道，red green blue红绿蓝
"""
#导包
import torch
import matplotlib.pyplot as plt
#1.定义函数，绘制：全黑，全白图
def dm01():
    #1.1定义全黑图片，也就是每个像素点都是0
    #HWC是：H 高度 W 宽度 C 通道数
    img1=torch.zeros(size=(200,200,3))
    print(f'img1:{img1}')
    #1.2绘制图片
    plt.imshow(img1)
    # plt.axis('off')  # 去除坐标轴
    plt.show()
    #定义全白图片，也就是每个像素点都是255
    img2=torch.full(size=(200,200,3),fill_value=255)
    print(f'img2:{img2}')
    #2.1绘制图片
    plt.imshow(img2)
    # plt.axis('off')  # 去除坐标轴
    plt.show()

#2.定义函数，加载图片
def dm02():
    #2.1加载图片
    img3=plt.imread('./data/img.jpg')
    print(f'img3:{img3}')
    #2.2保存图片
    plt.imsave('./data/img_save.jpg',img3)
    #2.3展示图片
    plt.imshow(img3)
    plt.axis('off')  # 去除坐标轴
    plt.show()





#测试函数
if __name__ == '__main__':
    # dm01()
    dm02()


