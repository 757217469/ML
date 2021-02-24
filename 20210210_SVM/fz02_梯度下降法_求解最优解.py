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


# 导函数
def h1(x):
    return x - .25


# 使用梯度下降法求解
GD_X = []
GD_Y = []
x = 4
alpha = .5
f_change = f1(x)
f_current = f_change
GD_X.append(x)
GD_Y.append(f_current)
iter_num = 0
while f_change > 1e-10 and iter_num < 100:
    iter_num += 1
    x = x - alpha * h1(x)
    tmp = f1(x)
    f_change = np.abs(f_current - tmp)
    f_current = tmp
    GD_X.append(x)
    GD_Y.append(f_current)
print(u'最终结果为:(%.5f,%.5f)' % (x, f_current))
print(u'迭代过程中X的取值, 迭代次数:%d' % iter_num)
print(GD_X)

# 构建数据
X = np.arange(-4, 4.5, .05)
Y = np.array(list(map(lambda t: f1(t), X)))

# 画图
plt.figure(facecolor='w')
plt.plot(X, Y, 'r-', linewidth=2)
plt.plot(GD_X, GD_Y, 'bo--', linewidth=2)
plt.title(u'函数$y=0.5 * (θ - 0.25)^2$; \n学习率:%.3f; 最终解:(%.3f,%.3f); 迭代次数:%d' % (alpha, x, f_current, iter_num))
plt.show()


# 二维原始图像
def f2(x, y):
    return .6 * (x + y) ** 2 - x * y


# 导函数
def hx2(x, y):
    return .6 * 2 * (x + y) - y


def hy2(x, y):
    return .6 * 2 * (x + y) - x


# 使用梯度下降法求解
GD_X1 = []
GD_X2 = []
GD_Y = []
x1 = 4
x2 = 4
alpha = .5
f_change = f2(x1, x2)
f_current = f_change
GD_X1.append(x1)
GD_X2.append(x2)
GD_Y.append(f_current)
iter_num = 0
while f_change > 1e-10 and iter_num < 100:
    iter_num += 1
    prex1 = x1
    prex2 = x2
    x1 = x1 - alpha * hx2(prex1, prex2)
    x2 = x2 - alpha * hy2(prex1, prex2)

    tmp = f2(x1, x2)
    f_change = np.abs(f_current - tmp)
    f_current = tmp
    GD_X1.append(x1)
    GD_X2.append(x2)
    GD_Y.append(f_current)
print(u'最终结果为:(%.5f,%.5f,%.5f' % (x1, x2, f_current))
print(u'迭代过程中X的取值, 迭代次数:%d' % iter_num)
print(GD_X1)

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
ax.plot(GD_X1, GD_X2, GD_Y, 'bo--')
ax.set_title(
    u'函数$y=0.6 * (θ1+θ2)^2 - θ1 * θ2$;\n学习率:%.3f; 最终解:(%.3f,%.3f,%.3f); 迭代次数:%d' % (alpha, x1, x2, f_current, iter_num))
plt.show()


# 二维原始图像
def f2_2(x, y):
    return .15 * (x + .5) ** 2 + .25 * (y - .25) ** 2 + .35 * (1.5 * x - .2 * y + .35) ** 2


# 偏导数
def hx2_2(x, y):
    return .15 * 2 * (x + .5) + .25 * 2 * (1.5 * x - .2 * y + .35) * 1.5


def hy2_2(x, y):
    return .25 * 2 * (y - .25) - .25 * 2 * (1.5 * x - .2 * y + .35) * .2


# 使用梯度下降法求解
GD_X1 = []
GD_X2 = []
GD_Y = []
x1 = 4
x2 = 4
alpha = .5
f_change = f2_2(x1, x2)
f_current = f_change
GD_X1.append(x1)
GD_X2.append(x2)
GD_Y.append(f_current)
iter_num = 0
while f_change > 1e-10 and iter_num < 100:
    iter_num += 1
    prex1 = x1
    prex2 = x2
    x1 = x1 - alpha * hx2_2(prex1, prex2)
    x2 = x2 - alpha * hy2_2(prex1, prex2)

    tmp = f2_2(x1, x2)
    f_change = np.abs(f_current - tmp)
    f_current = tmp
    GD_X1.append(x1)
    GD_X2.append(x2)
    GD_Y.append(f_current)
print(u'最终结果为:(%.5f,%.5f,%.5f)' % (x1, x2, f_current))
print(u'迭代过程中X的取值, 迭代次数:%d' % iter_num)
print(GD_X1)

# 构建数据
X1 = np.arange(-4, 4.5, .2)
X2 = np.arange(-4, 4.5, .2)
X1, X2 = np.meshgrid(X1, X2)
Y = np.array(list(map(lambda t: f2_2(t[0], t[1]), zip(X1.flatten(), X2.flatten()))))
Y.shape = X1.shape

# 画图
fig = plt.figure(facecolor='w')
ax = Axes3D(fig)
ax.plot_surface(X1, X2, Y, rstride=1, cstride=1, cmap=plt.cm.jet)
ax.plot(GD_X1, GD_X2, GD_Y, 'ko--')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title(u'函数:\n学习率:%.3f; 最终解:(%.3f, %.3f, %.3f);迭代次数:%d' % (alpha, x1, x2, f_current, iter_num))
plt.show()
