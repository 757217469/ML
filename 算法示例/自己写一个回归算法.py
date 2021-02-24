import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

## 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

X = np.logspace(0, 10, 50)
# Y = 1 / 3 * X ** 2 + 2 * X + 5
Y = 1/2 * X + 3
X = np.mat(X)
Y = np.mat(Y)
X.shape = -1, 1
Y.shape = -1, 1

theta = (X.T * X).I * X.T * Y
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
X_test = np.mat(X_test)
X_test.shape = -1, 1
Y_pre = X_test * theta

x = np.arange(len(X_test))
plt.figure(facecolor='w')
plt.plot(x, Y_test, 'r-', linewidth=5, label=u'真实值')
plt.plot(x, Y_pre, 'g-', linewidth=2, label=u'预测值')
plt.title(u'线性回归预测值和真实值对比', fontsize=20)
plt.grid(b=True)
plt.show()
