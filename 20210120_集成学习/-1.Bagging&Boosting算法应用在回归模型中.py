# coding=utf-8

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from sklearn.linear_model import LinearRegression, Ridge, LassoCV, ElasticNetCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor

# 设置字符集, 防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')


def notEmpty(s):
    return s != ''


## 加载数据
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
path = "datas/boston_housing.data"
# 由于数据文件格式不统一, 所以读取的时候, 先按照一行一个字段属性读取数据, 然后再按照每行数据进行处理
fd = pd.read_csv(path, header=None)
data = np.empty((len(fd), 14))

for i, d in enumerate(fd.values):
    d = map(float, filter(notEmpty, d[0].split(' ')))
    data[i] = list(d)
x, y = np.split(data, (13,), axis=1)
print(x[0:5])
y = y.ravel()
print(y[:5])
ly = len(y)
print(y.shape)
print('样本数据量:%d, 特征个数: %d' % x.shape)
print('target样本数据量:%d' % y.shape[0])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=28)

# 线性回归模型
lr = Ridge(alpha=.1)
lr.fit(x_train, y_train)
print('训练集上R^2:%.5f' % lr.score(x_train, y_train))
print('测试集上R^2:%.5f' % lr.score(x_test, y_test))

# 使用AdaBoostRegressor
adr = AdaBoostRegressor(LinearRegression(), n_estimators=100, learning_rate=.001, random_state=14)
adr.fit(x_train, y_train)
print('训练集上R^2:%.5f' % adr.score(x_train, y_train))
print('测试集上R^2:%.5f' % adr.score(x_test, y_test))

# 使用GBDT: GBDT模型只支持CART模型
gbdt = GradientBoostingRegressor(n_estimators=100, learning_rate=.01, random_state=14)
gbdt.fit(x_train, y_train)
print('训练集上R^2:%.5f' % gbdt.score(x_train, y_train))
print('测试集上T^2:%.5f' % gbdt.score(x_test, y_test))
