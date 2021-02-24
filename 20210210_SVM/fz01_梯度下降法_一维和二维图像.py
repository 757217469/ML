# coding=utf-8
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False


# 一维原始图像
def f1(x):
    return .5 * (x - .25) ** 2


# 构建数据
X = np.arange(-4, 4.5, .05)
Y = np.array(list(map(lambda t: f1(t), X)))

# 画图
plt.figure(facecolor='w')
plt.plot(X, Y, 'r-', linewidth=2)
plt.title(u'函数$0.5 * (θ - 0.25)^2$')
plt.show()


# 二维原始图像
def f2(x, y):
    return .6 * (x + y) ** 2 - x * y


# 构建数据
X1 = np.arange(-4, 4.5, .2)
X2 = np.arange(-4, 4.5, .2)
X1, X2 = np.meshgrid(X1, X2)
Y = np.array(list(map(lambda t: f2(t[0], t[1]), zip(X1.flatten(), X2.flatten()))))
Y.shape = X1.shape

# 画图
fig = plt.figure(facecolor='w')
ax = Axes3D(fig)
ax.plot_surface(X1, X2, Y, rstride=1, cstride=1, cmap=plt.cm.jet)
ax.set_title(u'函数$y=0.6 * (θ1 + θ2)^2 - θ1 * θ2$')
plt.show()
