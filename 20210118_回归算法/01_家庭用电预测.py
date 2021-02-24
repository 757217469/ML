# coding=utf-8
# 现行回归算法, 时间与功率&功率与电流之间的关系
# 引入所需要的全部包
# 数据划分的类
from sklearn.model_selection import train_test_split
# 线性回归的类
from sklearn.linear_model import LinearRegression
# 数据标准化
from sklearn.preprocessing import StandardScaler

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import time

mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# 加载数据
path1 = 'datas/household_power_consumption_1000.txt'
df = pd.read_csv(path1, sep=';', low_memory=False)

# 异常数据处理(异常数据过滤)
new_df = df.replace('?', np.nan)
# 只要有一个数据为空, 就进行行删除操作
datas = new_df.dropna(axis=0, how='any')


# 创建一个时间函数格式化字符串
def data_format(dt):
    t = time.strptime(' '.join(dt), '%d/%m/%Y %H:%M:%S')
    return (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)


## 需求: 构建时间和功率之间的映射关系, 可以认为: 特征属性为时间; 目标属性为功率值
# 获取x和y变量, 并将时间转换为数值型连续变量
X = datas.iloc[:, 0:2]
X = X.apply(lambda x: pd.Series(data_format(x)), axis=1)  # 以行为单位
Y = datas['Global_active_power']

# 对数据集进行测试集合训练集划分
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)

# 数据标准化
# StandardScaler: 将数据转换为标准差为1的数据集(有一个数据的映射)
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# 模型训练
lr = LinearRegression(fit_intercept=True)
lr.fit(X_train, Y_train)
y_predict = lr.predict(X_test)
print('训练集上R2: ', lr.score(X_train, Y_train))
print('测试集上R2: ', lr.score(X_test, Y_test))
mse = np.average((Y_test - y_predict) ** 2)
rmse = np.sqrt(mse)

print('模型的系数(θ): ', lr.coef_)
print('模型的截距: ', lr.intercept_)

# 模型保存/持久化
from sklearn.externals import joblib

# 保存模型要求给定的文件所在的文件夹存在
joblib.dump(ss, 'result/data_ss.model')
joblib.dump(lr, 'result/data_lr.model')

# 加载模型
ss3 = joblib.load('result/data_ss.model')
lr3 = joblib.load('result/data_lr.model')

# 使用加载的模型进行预测
data1 = [[2006, 12, 17, 12, 25, 0]]
data1 = ss3.transform(data1)
print(data1)
lr3.predict(data1)

t = np.arange(len(X_test))
plt.figure(facecolor='w')
plt.plot(t, Y_test, 'r-', linewidth=2, label='真实值')
plt.plot(t, y_predict, 'g-', linewidth=2, label='预测值')
plt.legend(loc='lower right')
plt.title('线性回归预测时间和功率之间的关系', fontsize=20)
plt.grid(b=True)
plt.show()

# 功率和电流之间的关系
X = datas.iloc[:, 2:4]
Y2 = datas.iloc[:, 5]

# 数据分割
X2_train, X2_test, Y2_train, Y2_test = train_test_split(X, Y2, test_size=0.2, random_state=0)

# 数据归一化
scaler2 = StandardScaler()
X2_train = scaler2.fit_transform(X2_train)
X2_test = scaler2.transform(X2_test)

# 模型训练
lr2 = LinearRegression()
lr2.fit(X2_train, Y2_train)

Y2_predict = lr2.predict(X2_test)

# 模型评估
print('电流预测准确率: ', lr2.score(X2_test, Y2_test))
print('电流参数: ', lr2.coef_)

# 绘图
t = np.arange(len(X2_test))
plt.figure(facecolor='w')
plt.plot(t, Y2_test, 'r-', linewidth=2, label=u'真实值')
plt.plot(t, Y2_predict, 'g-', linewidth=2, label=u'预测值')
plt.legend(loc='lower right')
plt.title(u'线性回归预测功率与电流之间的关系', fontsize=20)
plt.grid(b=True)
plt.show()
