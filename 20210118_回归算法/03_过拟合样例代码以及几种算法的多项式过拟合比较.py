# coding=utf-8
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import sklearn
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

# 设置字符集, 防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# 创建模拟数据
np.random.seed(100)
np.set_printoptions(linewidth=1000, suppress=True)
N = 10
# np.random.randn(N) 服从标准正态分布
x = np.linspace(0, 6, N) + np.random.randn(N)
y = 1.8 * x ** 3 + x ** 2 - 14 * x - 7 + np.random.randn(N)
## 将其设置为矩阵
x.shape = -1, 1
y.shape = -1, 1


## RidgeCV 和Ridge的区别是: 前者可以进行交叉验证
models = [
    Pipeline([
        ('Poly', PolynomialFeatures(include_bias=False)),
        ('Linear', LinearRegression(fit_intercept=False))
    ]),
    Pipeline([
        ('Poly', PolynomialFeatures(include_bias=False)),
        # alpha 给定的是Ridge算法中, L2正则项的权重值
        # alphas 是给定CV交叉验证过程中, Ridge算法的alpha参数值的取值范围
        ('Linear', RidgeCV(alphas=np.logspace(-3, 2, 50), fit_intercept=False))
    ]),
    Pipeline([
        ('Poly', PolynomialFeatures(include_bias=False)),
        ('Linear', LassoCV(alphas=np.logspace(0, 1, 10), fit_intercept=False))
    ]),
    Pipeline([
        ('Poly', PolynomialFeatures(include_bias=False)),
        ('Linear', ElasticNetCV(alphas=np.logspace(0, 1, 10), l1_ratio=[.1, .5, .7, .9, .95, 1], fit_intercept=False))
    ])
]
'''
# 线性模型过拟合图形识别
plt.figure(facecolor='w')
degree = np.arange(1, N, 4)
dm = degree.size
colors = []
for c in np.linspace(16711680, 255, dm):
    colors.append('#%06x' % int(c))

model = models[0]
for i, d in enumerate(degree):
    plt.subplot(int(np.ceil(dm / 2.0)), 2, i + 1)
    plt.plot(x, y, 'ro', ms=5, zorder=N)

    # 设置阶数
    model.set_params(Poly__degree=d)
    # 模型训练
    model.fit(x, y.ravel())

    lin = model.get_params()['Linear']
    output = u'%d阶, 系数为: ' % (d)
    # 判断lin对象中是否有对应的属性
    if hasattr(lin, 'alpha_'):
        idx = output.find(u'系数')
        output = output[:idx] + (u'alpha=%.6f, ' % lin.alpha_) + output[idx:]
    if hasattr(lin, 'l1_ratio_'):
        idx = output.find(u'系数')
        output = output[:idx] + (u'l1_ratio=%.6f,' % lin.l1_ratio_) + output[idx:]
    print(output, lin.coef_.ravel())

    x_hat = np.linspace(x.min(), x.max(), num=100)
    x_hat.shape = -1, 1
    y_hat = model.predict(x_hat)
    s = model.score(x, y)

    z = N - 1 if (d == 2) else 0
    label = u'%d 阶,正确率%.3f' % (d, s)
    plt.plot(x_hat, y_hat, color=colors[i], lw=2, alpha=.75, label=label, zorder=z)
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.xlabel('X', fontsize=16)
    plt.ylabel('Y', fontsize=16)

plt.tight_layout(1, rect=(0, 0, 1, .95))
plt.suptitle(u'线性回归过拟合显示', fontsize=22)
plt.show()
'''
# 线性回归, LASSO回归, Ridge回归, ElasticNet比较
plt.figure(facecolor='w')
degree = np.arange(1, N, 2)
dm = degree.size
colors = []
for c in np.linspace(16711680, 255, dm):
    colors.append('#%06x' % int(c))
titles = [u'线性回归', u'Ridge回归', u'Lasso回归', u'ElasticNet']

for t in range(4):
    model = models[t]
    plt.subplot(2, 2, t + 1)
    plt.plot(x, y, 'ro', ms=10, zorder=N)

    # 遍历不同的多项式的阶, 看不同阶的情况下, 模型的效果
    for i, d in enumerate(degree):
        model.set_params(Poly__degree=d)
        model.fit(x, y.ravel())
        lin = model.get_params()['Linear']
        output = u'%s:%d阶,系数为: ' % (titles[t], d)
        if hasattr(lin, 'alpha_'):  # 判断lin这个模型中是否有alpha_这个属性
            idx = output.find(u'系数')
            output = output[:idx] + (u'alpha=%.6f, ' % lin.alpha_) + output[idx:]
        if hasattr(lin, 'l1_ratio_'):  # 判断lin这个模型中是否有l1_ratio_这个属性
            idx = output.find(u'系数')
            output = output[:idx] + (u'l1_ratio=%.6f, ' % lin.l1_ratio_) + output[idx:]
            # line.coef_：获取线性模型的参数列表，也就是我们ppt中的theta值，ravel()将结果转换为1维数据
        print(output, lin.coef_.ravel())

        x_hat = np.linspace(x.min(), x.max(), num=100)
        x_hat.shape = -1, 1
        y_hat = model.predict(x_hat)
        s = model.score(x, y)

        z = N + 1 if (d == 5) else 0
        label = u'%d阶, 正确率=%.3f' % (d, s)
        plt.plot(x_hat, y_hat, color=colors[i], lw=2, alpha=.75, label=label, zorder=z)

    plt.legend(loc='upper left')
    plt.grid(True)
    plt.title(titles[t])
    plt.xlabel('X', fontsize=16)
    plt.ylabel('Y', fontsize=16)
plt.tight_layout(1, rect=(0, 0, 1, 0.95))
plt.suptitle(u'各种不同线性回归过拟合显示', fontsize=22)
plt.show()
