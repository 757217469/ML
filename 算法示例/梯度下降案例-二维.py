from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# 解决中文显示问题
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False


## 原函数
def f(x, y):
    return x ** 2 + y ** 2


# 偏函数:
def h(t):
    return 2 * t


X = []
Y = []
Z = []

x = 2
y = 2
f_change = x ** 2 + y ** 2
f_current = f(x, y)
step = 0.5
X.append(x)
Y.append(y)
Z.append(f_current)
iter_count = 0
while f_change > 1e-10:
    x = x - step * h(x)
    y = y - step * h(y)
    f_change = np.abs(f_current - f(x, y))
    f_current = f(x, y)
    X.append(x)
    Y.append(y)
    Z.append(f_current)
    iter_count += 1
print(f_change)
fig = plt.figure()
# ax = Axes3D(fig)   # 与上面等价
ax = fig.add_subplot(1, 1, 1, projection='3d')
X2 = np.arange(-2, 2, 0.2)
Y2 = np.arange(-2, 2, 0.2)
X2, Y2 = np.meshgrid(X2, Y2)
Z2 = X2 ** 2 + Y2 ** 2

ax.plot_surface(X2, Y2, Z2, rstride=1, cstride=1, cmap='rainbow', zorder=20)
ax.plot(X, Y, Z, 'ro--', zorder=10)

# elev,azim
ax.view_init(51.25, -164.5)

ax.set_title(u'梯度下降法求解, 最终解为: x=%.2f,y=%.2f,z=%.2f,迭代次数为 %d' % (x, y, f_current, iter_count))
plt.show()
