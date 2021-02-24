# coding=utf-8
# 葡萄酒质量预测
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import sklearn
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, label_binarize
from sklearn import metrics
from sklearn.model_selection import train_test_split

mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# 读取数据
path1 = "datas/winequality-red.csv"
df1 = pd.read_csv(path1, sep=';')
df1['type'] = 1

path2 = 'datas/winequality-white.csv'
df2 = pd.read_csv(path2, sep=';')
df2['type'] = 2

# 合并df
df = pd.concat([df1, df2], axis=0)

## 自变量名称
names = ["fixed acidity", "volatile acidity", "citric acid",
         "residual sugar", "chlorides", "free sulfur dioxide",
         "total sulfur dioxide", "density", "pH", "sulphates",
         "alcohol", "type"]

## 因变量名称
quality = "quality"

# 异常数据处理
datas = df.replace('?', np.nan).dropna(how='any', axis=0)
print('原始数据条数:%d; 异常数据处理后数据条数:%d' % (len(df), len(datas)))

# 提取自变量和因变量
X = datas[names]
Y = datas[quality]

# 数据分割
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.23, random_state=0)

# 2.数据格式化(归一化)
# 将数据缩放到[0,1]
ss = MinMaxScaler()
X_train = ss.fit_transform(X_train)
Y_train.value_counts()

# 模型构建及训练
## penalty: 过拟合解决参数,l1或l2
# solver: 参数优化的方式
# 当penalty 为l1的时候, 参数只能是liblinear(坐标轴下降法)
# 当penalty 为l2的时候, 参数可以使: lbfgs(拟牛顿法), newton-cg(牛顿法变种)

# multi_class: 分类方式参数; 参数可选: ovr(默认),multinomial;这两种方式在二元分类问题中效果是一样的, 在多元分类问题中效果不一样
# over: one-vs-rest, 对于多元分类问题,可先将其看成二元分类, 分类完成后,再迭代对其中一类继续进行二元分类
# multinomial: many-vs-many(MVM),对于多元分类问题,如果模型有T类, 我们每次在所有的T类样本里面选择两类样本出来
# 不妨记为T1类和T2类, 把所有的输出为T1和T2的样本放在一起,把T1作为正例,T2作为负例,
# 进行二元逻辑回归, 得到模型参数. 我们一共需要T(T-1)/2次分类

# class_weight: 特征权重参数
# Softmax算法相对于Logistic算法来讲, 在sklearn中体现的代码形式来讲, 主要只是参数的不同
# Logistic算法回归(二分类): 使用的是ovr, 如果是softmax回归,建议使用multinomial

lr = LogisticRegressionCV(fit_intercept=True, Cs=np.logspace(-5, 1, 100), multi_class='multinomial', penalty='l2',
                          solver='lbfgs')
lr.fit(X_train, Y_train)

# 4.模型效果获取
r = lr.score(X_train, Y_train)
print('R值: ', r)
print('特征系数花比率: %.2f%%' % (np.mean(lr.coef_.ravel() == 0) * 100))
print('参数: ', lr.coef_)
print('截距: ', lr.intercept_)
print('概率: ', lr.predict_proba(X_test))
print(lr.predict_proba(X_test).shape)

# 数据预测
# a.预测数据格式化(归一化)
X_test = ss.transform(X_test)
Y_predict = lr.predict(X_test)

x_len = range(len(X_test))
plt.figure(figsize=(14, 7), facecolor='w')
plt.plot(x_len, Y_test, 'ro', markersize=8, zorder=3, label=u'真实值')
plt.plot(x_len, Y_predict, 'go', markersize=12, zorder=2, label=u'预测值,$R^2$=%.3f' % lr.score(X_train, Y_train))
plt.legend(loc='upper left')
plt.xlabel(u'数据编号', fontsize=18)
plt.ylabel(u'葡萄酒质量', fontsize=18)
plt.title(u'葡萄酒质量预测统计', fontsize=20)
plt.show()
