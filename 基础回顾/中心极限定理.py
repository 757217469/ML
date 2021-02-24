import random
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

random.seed(28)
u"""
随机的抛六面的骰子, 计算三次的点数的和, 三次点数的和其实就是一个事件A
==> 事件A的发生属于什么分布??
==> A = x1+x2+x3, 其中x1,x2,x3是分别三次的抛骰子的点数
根据中心极限定理,由于x1,x2,x3属于独立同分布的,所以说最终的事件A属于高斯分布
"""


def generate_random_int():
    """随机产生一个[1,6]的数字, 表示的是一个六面骰子的结果"""
    return random.randint(1, 6)


def generate_mean(n):
    """计算返回n次抛六面骰子的和结果"""
    return np.sum([generate_random_int() for i in range(n)])


if __name__ == '__main__':
    # 进行A事件多少次
    number1 = 10000000
    # 表示每次A事件抛几次骰子
    number2 = 3

    # 进行number1次事件A的操作, 每次事件A都进行number2次抛骰子
    keys = [generate_mean(number2) for i in range(number1)]

    # 统计每个和数字出现的次数, eg: 和为3 的出现多少次,和为10 的出现多少次
    result = {}
    for key in keys:
        count = 1
        if key in result:
            count += result[key]
            result[key] = count
    # 获取x和y
    x = sorted(np.unique(list(result.keys())))
    y = []
    for key in x:
        # 将出现的次数进行一个辈分比的计算
        y.append(result[key] / number1)
    # 画图
    plt.plot(x, y, 'b-')
    plt.xlim(x[0] - 1, x[-1] + 1)
    plt.grid(True)
    plt.show()
