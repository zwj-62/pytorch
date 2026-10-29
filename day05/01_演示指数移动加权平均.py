"""
案例：
    演示近30天的天气分布情况

结论：
    针对baita（调节权重系数）来讲。其值越大说明其值越依赖指数加权平均，越不依赖本地的梯度值，数值就越平缓

"""
"""演示指数移动加权平均（EWA）：对比原始温度与不同 beta 下的平滑曲线"""
import torch
import matplotlib.pyplot as plt

ELEMENT_NUMBER = 30

def dm01():
    """原始温度（不加权）"""
    torch.manual_seed(0)
    temperature = torch.randn(size=[ELEMENT_NUMBER,]) * 10
    days = torch.arange(1, ELEMENT_NUMBER + 1, 1)
    plt.figure()
    plt.plot(days, temperature, color='red')
    plt.scatter(days, temperature)
    plt.title('raw temperature')
    plt.show()

def dm02(beta=0.9):
    """指数移动加权平均：v_t = beta * v_{t-1} + (1 - beta) * theta_t"""
    torch.manual_seed(0)
    temperature = torch.randn(size=[ELEMENT_NUMBER,]) * 10

    exp_weight_avg = []
    for idx, temp in enumerate(temperature, 1):
        if idx == 1:
            exp_weight_avg.append(temp) # 第一个元素的 EWA 取自身
        else:
            exp_weight_avg.append(exp_weight_avg[-1] * beta + temp * (1 - beta))

    days = torch.arange(1, ELEMENT_NUMBER + 1, 1)
    plt.figure()
    plt.scatter(days, temperature, color='red', label='raw')
    plt.plot(days, exp_weight_avg, color='blue', label=f'EWA beta={beta}')
    plt.legend()
    plt.title(f'EWA, beta={beta}')
    plt.show()

if __name__ == '__main__':
    dm01()
    dm02(0.5)
    dm02(0.9)