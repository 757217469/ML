# coding=utf-8
import time
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'
path = './datas/iris.data'
data = pd.read_csv(path, header=None)
x, y = data[list(range(4))], data[4]
y = pd.Categorical(y).codes
x = x[[0, 1]]
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=28, train_size=.6)

svm1 = SVC(C=.1, kernel='rbf')
svm2 = SVC(C=1, kernel='rbf')
svm3 = SVC(C=10, kernel='rbf')
svm4 = SVC(C=100, kernel='rbf')
svm5 = SVC(C=500, kernel='rbf')
svm6 = SVC(C=100000, kernel='rbf')
# C越大, 泛化能力越差,会出现过拟合的问题
# C越小, 繁华能力越好, 但是容易出现欠拟合的问题
# 模型训练
t0 = time.time()
svm1.fit(x_train, y_train)
t1 = time.time()
svm2.fit(x_train, y_train)
t2 = time.time()
svm3.fit(x_train, y_train)
t3 = time.time()
svm4.fit(x_train, y_train)
t4 = time.time()
svm5.fit(x_train, y_train)
t5 = time.time()
svm6.fit(x_train, y_train)
t6 = time.time()

svm1_score1 = accuracy_score(y_train, svm1.predict(x_train))
svm1_score2 = accuracy_score(y_test, svm1.predict(x_test))

svm2_score1 = accuracy_score(y_train, svm2.predict(x_train))
svm2_score2 = accuracy_score(y_test, svm2.predict(x_test))

svm3_score1 = accuracy_score(y_train, svm3.predict(x_train))
svm3_score2 = accuracy_score(y_test, svm3.predict(x_test))

svm4_score1 = accuracy_score(y_train, svm4.predict(x_train))
svm4_score2 = accuracy_score(y_test, svm4.predict(x_test))

svm5_score1 = accuracy_score(y_train, svm5.predict(x_train))
svm5_score2 = accuracy_score(y_test, svm5.predict(x_test))

svm6_score1 = accuracy_score(y_train, svm6.predict(x_train))
svm6_score2 = accuracy_score(y_test, svm6.predict(x_test))

x_tmp = [0, 1, 2, 3, 4, 5]
t_score = [t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5]
y_score1 = [svm1_score1, svm2_score1, svm3_score1, svm4_score1, svm5_score1, svm6_score1]
y_score2 = [svm1_score2, svm2_score2, svm3_score2, svm4_score2, svm5_score2, svm6_score2]

plt.figure(facecolor='w', figsize=(12, 6))
plt.subplot(121)
plt.plot(x_tmp, y_score1, 'r-', lw=2, label=u'训练集准确率')
plt.plot(x_tmp, y_score2, 'g-', lw=2, label=u'测试集准确率')
plt.xlim(-.3, 5.3)
plt.ylim(np.min((np.min(y_score1), np.min(y_score2))) * .9, np.max((np.max(y_score1), np.max(y_score2))) * 1.1)
plt.legend(loc='lower left')
plt.title(u'模型预测准确率', fontsize=13)
plt.xticks(x_tmp, [u'C=0.1', u'C=1', u'C=10', u'C=100', u'C=500', u'C=100000'], rotation=0)
plt.grid(b=True)

plt.subplot(122)
plt.plot(x_tmp, t_score, 'b-', lw=2, label=u'模型训练时间')
plt.title(u'模型训练耗时', fontsize=13)
plt.xticks(x_tmp, [u'C=0.1', u'C=1', u'C=10', u'C=100', u'C=500', u'C=100000'], rotation=0)
plt.grid(b=True)

plt.suptitle(u'鸢尾花数据SVM分类器不同内核函数模型比较', fontsize=16)
plt.show()

N = 500
x1_min, x2_min = x.min()
x1_max, x2_max = x.max()

t1 = np.linspace(x1_min, x1_max, N)
t2 = np.linspace(x2_min, x2_max, N)
x1, x2 = np.meshgrid(t1, t2)
grid_show = np.dstack((x1.flat, x2.flat))[0]

# 获取各个不同算法的测试值
svm1_grid_hat = svm1.predict(grid_show).reshape(x1.shape)
svm2_grid_hat = svm2.predict(grid_show).reshape(x1.shape)
svm3_grid_hat = svm3.predict(grid_show).reshape(x1.shape)
svm4_grid_hat = svm4.predict(grid_show).reshape(x1.shape)
svm5_grid_hat = svm5.predict(grid_show).reshape(x1.shape)
svm6_grid_hat = svm6.predict(grid_show).reshape(x1.shape)

cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
plt.figure(facecolor='w', figsize=(14, 7))

plt.subplot(231)
plt.pcolormesh(x1, x2, svm1_grid_hat, cmap=cm_light)
plt.scatter(x[0], x[1], c=y, edgecolors='k', s=50, cmap=cm_dark)
plt.xlabel(iris_feature[0], fontsize=13)
plt.ylabel(iris_feature[1], fontsize=13)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title(u'C=0.1', fontsize=15)
plt.grid(b=True, ls=':')
plt.tight_layout(pad=1.5)

plt.subplot(232)
plt.pcolormesh(x1, x2, svm2_grid_hat, cmap=cm_light)
plt.scatter(x[0], x[1], c=y, edgecolors='k', s=50, cmap=cm_dark)
plt.xlabel(iris_feature[0], fontsize=13)
plt.ylabel(iris_feature[1], fontsize=13)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title(u'C=1', fontsize=15)
plt.grid(b=True, ls=':')
plt.tight_layout(pad=1.5)
plt.subplot(233)
## 区域图
plt.pcolormesh(x1, x2, svm3_grid_hat, cmap=cm_light)
## 所以样本点
plt.scatter(x[0], x[1], c=y, edgecolors='k', s=50, cmap=cm_dark)      # 样本
## 测试数据集
plt.scatter(x_test[0], x_test[1], s=120, facecolors='none', zorder=10)     # 圈中测试集样本
## lable列表
plt.xlabel(iris_feature[0], fontsize=13)
plt.ylabel(iris_feature[1], fontsize=13)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title(u'C=10', fontsize=15)
plt.grid(b=True, ls=':')
plt.tight_layout(pad=1.5)

plt.subplot(234)
## 区域图
plt.pcolormesh(x1, x2, svm4_grid_hat, cmap=cm_light)
## 所以样本点
plt.scatter(x[0], x[1], c=y, edgecolors='k', s=50, cmap=cm_dark)      # 样本
## 测试数据集
plt.scatter(x_test[0], x_test[1], s=120, facecolors='none', zorder=10)     # 圈中测试集样本
## lable列表
plt.xlabel(iris_feature[0], fontsize=13)
plt.ylabel(iris_feature[1], fontsize=13)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title(u'C=100', fontsize=15)
plt.grid(b=True, ls=':')
plt.tight_layout(pad=1.5)

plt.subplot(235)
## 区域图
plt.pcolormesh(x1, x2, svm5_grid_hat, cmap=cm_light)
## 所以样本点
plt.scatter(x[0], x[1], c=y, edgecolors='k', s=50, cmap=cm_dark)      # 样本
## 测试数据集
plt.scatter(x_test[0], x_test[1], s=120, facecolors='none', zorder=10)     # 圈中测试集样本
## lable列表
plt.xlabel(iris_feature[0], fontsize=13)
plt.ylabel(iris_feature[1], fontsize=13)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title(u'C=500', fontsize=15)
plt.grid(b=True, ls=':')
plt.tight_layout(pad=1.5)

plt.subplot(236)
## 区域图
plt.pcolormesh(x1, x2, svm6_grid_hat, cmap=cm_light)
## 所以样本点
plt.scatter(x[0], x[1], c=y, edgecolors='k', s=50, cmap=cm_dark)      # 样本
## 测试数据集
plt.scatter(x_test[0], x_test[1], s=120, facecolors='none', zorder=10)     # 圈中测试集样本
## lable列表
plt.xlabel(iris_feature[0], fontsize=13)
plt.ylabel(iris_feature[1], fontsize=13)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title(u'C=10000', fontsize=15)
plt.grid(b=True, ls=':')
plt.tight_layout(pad=1.5)


plt.suptitle(u'鸢尾花数据SVM分类器不同C参数效果比较', fontsize=16)
plt.show()

