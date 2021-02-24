import random
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

# 解决中文显示问题
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 给定随机数的种子
random.seed(28)


def generate_random_int(n):
    """产生n个1-9的随机数"""
    return [random.randint(1, 9) for i in range(n)]


if __name__ == '__main__':
    number = 8000
    x = [i for i in range(number + 1) if i != 0]
    # 产生number个[1,9]的随机数
    total_random_int = generate_random_int(number)
    # 求n个[1,9]的随机数的均值,n=1,2,3,4,5......
    y = [np.mean(total_random_int[0:i + 1]) for i in range(number)]

    plt.plot(x, y, 'b-')
    plt.xlim(0, number)
    plt.grid(True)
    plt.show()
