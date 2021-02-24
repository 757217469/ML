# coding=utf-8
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
# 产生训练数据
X = 0.3 * np.random.randn(100, 2)
X_train = np.r_[X + 2, X - 2]
# 产生测试数据
X = 0.3 * np.random.randn(20, 2)
X_test = np.r_[X + 2, X - 2]
# 产生一些异常点数据
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

# 模型训练
clf = svm.OneClassSVM(nu=0.01, kernel='rbf', gamma=0.1)
clf.fit(X_train)

# 预测结果获取
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)
# 返回1表示属于这个类别,-1表示不属于这个类别
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size
n_error_outliers = y_pred_outliers[y_pred_outliers == -1].size

# 获取绘图的点信息
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 画图
plt.figure(facecolor='w')
plt.title('异常点检测')
# 画出区域图
'''
plt.contourf 与 plt.contour 区别：

f：filled，也即对等高线间的填充区域进行填充（使用不同的颜色）
contourf：将不会再绘制等高线（显然不同的颜色分界就表示等高线本身）
'''
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 9), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

# 画出点图
s = 40
b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, edgecolors='k')
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s, edgecolors='k')
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=s, edgecolors='k')

# 设置相关信息
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([a.collections[0], b1, b2, c],
           ['分割超平面', '训练样本', '测试样本', '异常点'],
           loc='upper left',
           prop=matplotlib.font_manager.FontProperties(size=11))
plt.xlabel('训练集错误率:%d/200 ; 测试集错误率: %d/40; 异常点错误率:%d/40' % (n_error_train, n_error_test, n_error_outliers))
plt.show()
