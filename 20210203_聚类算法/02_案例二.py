# coding=utf-8
# K-means算法和Mini Batch K-Means算法比较

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 初始化三个中心
centers = [[1, 1], [-1, -1], [1, -1]]
clusters = len(centers)
# 产生3000组二维的数据,中心点数是3, 标准差是7
X, Y = make_blobs(n_samples=3000, centers=centers, cluster_std=.7, random_state=28)

# 构建K-Means算法
k_means = KMeans(init='k-means++', n_clusters=clusters, random_state=28)
t0 = time.time()
k_means.fit(X)
km_batch = time.time() - t0
print('K-Means算法模型训练消耗时间:%.4fs' % km_batch)

# 构建Mini Batch K-Means 算法
batch_size = 100
mbk = MiniBatchKMeans(init='k-means++', n_clusters=clusters, batch_size=batch_size, random_state=28)
t0 = time.time()
mbk.fit(X)
mbk_batch = time.time() - t0
print('Mini Batch K-Means算法模型训练消耗时间:%.4fs' % mbk_batch)

km_y_hat = k_means.predict(X)
mbkm_y_hat = mbk.predict(X)

print(km_y_hat[:10])
print(mbkm_y_hat[:10])
print(k_means.cluster_centers_)
print(mbk.cluster_centers_)

# 获取聚类中心点并聚类中心点进行排序
k_means_cluster_centers = k_means.cluster_centers_
mbk_means_cluster_centers = mbk.cluster_centers_
print('K-Means算法聚类中心点:\ncenter=', k_means_cluster_centers)
print('Mini Batch K-Means算法聚类中心点:\ncenter=', mbk_means_cluster_centers)
# 方便后面画图
order = pairwise_distances_argmin(k_means_cluster_centers, mbk_means_cluster_centers)

plt.figure(figsize=(12, 6), facecolor='w')
plt.subplots_adjust(left=.05, right=.95, bottom=.05, top=.9)
cm = mpl.colors.ListedColormap(['#FFC2CC', '#C2FFCC', '#CCC2FF'])
cm2 = mpl.colors.ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# 子图1: 原始数据
plt.subplot(221)
plt.scatter(X[:, 0], X[:, 1], c=Y, s=6, cmap=cm, edgecolors='none')
plt.title(u'原始数据分布图')
plt.xticks(())
plt.yticks(())
plt.grid(True)

# 子图2:K-Means算法聚类结果图
plt.subplot(222)
plt.scatter(X[:, 0], X[:, 1], c=km_y_hat, s=6, cmap=cm, edgecolors='none')
plt.scatter(k_means_cluster_centers[:, 0], k_means_cluster_centers[:, 1], c=range(clusters), s=60, cmap=cm2,
            edgecolors='none')
plt.title(u'K-Means算法聚类结果图')
plt.xticks(())
plt.yticks(())
plt.text(-3.8, 3, 'train time: %.2fms' % (km_batch * 1000))
plt.grid(True)

# 子图3: Mini Batch K-Means 算法聚类结果图
plt.subplot(223)
plt.scatter(X[:, 0], X[:, 1], c=mbkm_y_hat, s=6, cmap=cm, edgecolors='none')
plt.scatter(mbk_means_cluster_centers[:, 0], mbk_means_cluster_centers[:, 1], c=range(clusters), cmap=cm2,
            edgecolors='none')
plt.title(u'Mini Batch K-Means 算法聚类结果图')
plt.xticks(())
plt.yticks(())
plt.text(-3.8, 3, 'train time: %.2fms' % (mbk_batch * 1000))
plt.grid(True)

#
different = list(map(lambda x: (x != 0) & (x != 1) & (x != 2), mbkm_y_hat))
for k in range(clusters):
    different += ((km_y_hat == k) != (mbkm_y_hat == order[k]))
identic = np.logical_not(different)
different_nodes = len(list(filter(lambda x: x, different)))

plt.subplot(224)
# 两者预测相同的
plt.plot(X[identic, 0], X[identic, 1], 'w', markerfacecolor='#bbbbbb', marker='.')
# 两者预测不相同的
plt.plot(X[different, 0], X[different, 1], 'w', markerfacecolor='m', marker='.')
plt.title(u'Mini Batch K-Means和K-means算法预测结果不同的点')
plt.xticks(())
plt.yticks(())
plt.text(-3.8, 2, 'different nodes: %d' % (different_nodes))
plt.show()
