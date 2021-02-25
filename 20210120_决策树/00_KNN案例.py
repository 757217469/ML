# coding=utf-8
# 鸢尾花数据分类
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from sklearn import metrics

mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# 数据加载
path = "datas/iris.data"
names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'cla']
df = pd.read_csv(path, header=None, names=names)
df['cla'].value_counts()
df.head()


def parseRecord(record):
    result = []
    r = zip(names, record)
    for name, v in r:
        if name == 'cla':
            if v == 'Iris-setosa':
                result.append(1)
            elif v == 'Iris-versicolor':
                result.append(2)
            else:
                result.append(3)
        else:
            result.append(float(v))
    return result


# 1.数据转换为数字以及分割
datas = df.apply(lambda x: pd.Series(parseRecord(x), index=names), axis=1)
# 异常数据删除
datas = datas.dropna(how='any')
# 数据分割
X = datas[names[:-1]]
Y = datas[names[-1]]

# 数据抽样(训练数据和测试数据分割)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.4, random_state=0)
print('原始数据条数: %d; 训练数据条数: %d; 特征个数: %d; 测试样本条数: %d' % (len(X), len(X_train), X_train.shape[1], X_test.shape[0]))

# KNN 算法实现
# a.模型构建
# 模型中介绍的K值: n_neighbors
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)

# b.模型效果输出
# 将正确的数据转换为矩阵形式
y_test_hot = label_binarize(Y_test, classes=(1, 2, 3))
# 得到预测属于某个类别的概率值
knn_y_score = knn.predict_proba(X_test)
# 计算roc值
knn_fpr, knn_tpr, knn_threshold = metrics.roc_curve(y_test_hot.ravel(), knn_y_score.ravel())
# 计算auc的值
knn_auc = metrics.auc(knn_fpr, knn_tpr)
print('KNN算法R2值: ', knn.score(X_train, Y_train))
print('KNN算法AUC值: ', knn_auc)

# c.模型预测
knn_y_predict = knn.predict(X_test)
# 画图2: 预测结果画图
x_test_len = range(len(X_test))
plt.figure(figsize=(12, 9), facecolor='w')
plt.plot(x_test_len, Y_test, 'ro', markersize=6, zorder=3, label=u'实际值')
plt.plot(x_test_len, knn_y_predict, 'yo', markersize=16, zorder=1,
         label=u'KNN算法预测值,$R^2$=%.3f' % knn.score(X_test, Y_test))
plt.legend(loc='lower right')
plt.xlabel(u'数据编号', fontsize=18)
plt.ylabel(u'种类', fontsize=18)
plt.title(u'鸢尾花数据分类', fontsize=20)
plt.show()
