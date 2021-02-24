from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np

# 解决中文显示问题
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False


## 原函数
def f(x):
    return x ** 2


# 导数
def h(x):
    return 2 * x


if __name__ == '__main__':
    X = []
    Y = []
    x = 2
    step = 0.8
    f_change = f(x)
    f_current = f(x)
    X.append(x)
    Y.append(f_current)
    while f_change > 1e-10:
        x = x - step * h(x)
        tmp = f(x)
        f_change = np.abs(f_current - tmp)
        f_current = tmp
        X.append(x)
        Y.append(f_current)
    fig = plt.figure()
    X2 = np.arange(-2.1, 2.15, 0.05)
    Y2 = X2 ** 2
    plt.plot(X2, Y2, '-', color='#666666', linewidth=2)
    plt.plot(X, Y, 'bo--')
    plt.title(u'$y-x^2$函数求解最小值,最终解为:x=%.2f,y=%.2f' % (x, f_current))
    plt.show()
