# coding=utf-8
import numpy as np
import sklearn
from sklearn.datasets import make_blobs  # 导入产生模拟数据的方法
from sklearn.cluster import KMeans

# 1.产生模拟数据
N = 1000
centers = 4
X, Y = make_blobs(n_samples=N, n_features=2, centers=centers, random_state=28)

# 2.模型构建
km = KMeans(n_clusters=centers, init='random', random_state=28)
km.fit(X)

# 预测
y_hat = km.predict(X[:10])
print(y_hat)

print('所有样本距离所属簇中心点的总距离和为:%.5f' % km.inertia_)
print('所有样本距离所属簇中心点的平均距离为:%.5f' % (km.inertia_ / N))

cluster_centers = km.cluster_centers_
print('所有的中心点聚类中心坐标:')
print(cluster_centers)

print('score其实就是所有样本点离所属簇中心点距离和的相反数')
print(km.score(X))
