# coding=utf-8
# 葡萄酒质量预测

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import sklearn
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')
mpl.rcParams['font.sans-serif'] = ['simHei']
mpl.rcParams['axes.unicode_minus'] = False

# 读取数据
path1 = "datas/winequality-red.csv"
df1 = pd.read_csv(path1, sep=';')
df1['type'] = 1  # 设置数据类型为红葡萄酒

path2 = "datas/winequality-white.csv"
df2 = pd.read_csv(path2, sep=';')
df2['type'] = 2  # 白葡萄酒

# 合并
df = pd.concat([df1, df2], axis=0)  # axis=0 按照行合并

## 自变量名称
names = ["fixed acidity", "volatile acidity", "citric acid",
         "residual sugar", "chlorides", "free sulfur dioxide",
         "total sulfur dioxide", "density", "pH", "sulphates",
         "alcohol", "type"]

## 因变量名称
quality = "quality"

names1 = []
# list(df)  生成columns组成的列表
for i in list(df):
    names1.append(i)

# 异常数据处理
new_df = df.replace('?', np.nan)
datas = new_df.dropna(how='any')

X = datas[names]
Y = datas[quality]
Y.ravel()

# 构建model
models = [
    Pipeline([
        ('ss', StandardScaler()),
        ('poly', PolynomialFeatures()),
        ('linear', LinearRegression())
    ]),
    Pipeline([
        ('ss', StandardScaler()),
        ('poly', PolynomialFeatures()),
        ('linear', RidgeCV(alphas=np.logspace(-4, 2, 20)))
    ]),
    Pipeline([
        ('ss', StandardScaler()),
        ('poly', PolynomialFeatures()),
        ('linear', LassoCV(alphas=np.logspace(-4, 2, 20)))
    ]),
    Pipeline([
        ('ss', StandardScaler()),
        ('poly', PolynomialFeatures()),
        ('linear', ElasticNetCV(alphas=np.logspace(-4, 2, 20), l1_ratio=np.linspace(0, 1, 5)))
    ])
]

plt.figure(figsize=(16, 8), facecolor='w')
titles = u'线性回归预测', u'Ridge回归预测', u'Lasso回归预测', u'ElasticNet预测'

# 将数据分为训练数据和测试数据
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.01, random_state=0)
ln_x_test = range(len(X_test))

# 给定阶以及颜色
d_pool = np.arange(1, 4, 1)
m = len(d_pool)
clrs = []
for c in np.linspace(5570560, 255, m):
    clrs.append('#%06x' % int(c))

for t in range(4):
    plt.subplot(2, 2, t + 1)
    model = models[t]
    plt.plot(ln_x_test, Y_test, c='r', lw=2, alpha=.75, zorder=10, label=u'真实值')
    for i, d in enumerate(d_pool):
        # 设置参数
        model.set_params(poly__degree=d)
        # train
        model.fit(X_train, Y_train)
        # 模型预测及计算R^2
        Y_pre = model.predict(X_test)
        R = model.score(X_train, Y_train)
        # 输出信息
        lin = model.get_params()['linear']
        output = u'%s:%d阶, 截距:%d, 系数: ' % (titles[t], d, lin.intercept_)
        print(output, lin.coef_)
        plt.plot(ln_x_test, Y_pre, c=clrs[i], lw=2, alpha=.75, zorder=i, label=u'%d阶预测值,$R^2$=%.3f' % (d, R))
    plt.legend(loc=('upper left'))
    plt.grid(True)
    plt.title(titles[t], fontsize=18)
    plt.xlabel('X', fontsize=16)
    plt.ylabel('Y', fontsize=16)
plt.suptitle(u'葡萄酒质量预测', fontsize=22)
plt.show()
