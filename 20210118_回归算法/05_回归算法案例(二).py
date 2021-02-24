# coding=utf-8
# 波士顿房屋租赁价格预测
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import sklearn
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')


def notEmpty(s):
    return s != ''


# 加载数据
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
path = "datas/boston_housing.data"
fd = pd.read_csv(path, header=None)
data = np.empty((len(fd), 14))
for i, d in enumerate(fd.values):
    d = map(float, filter(notEmpty, d[0].split(' ')))
    data[i] = list(d)

# 分割数据
x, y = np.split(data, (13,), axis=1)
print(x[:5])
y = y.ravel()
print(y[:5])
ly = len(y)
print(y.shape)
print('样本数据量: %d, 特征个数: %d' % x.shape)
print('target 样本数据量: %d' % y.shape[0])

models = [
    Pipeline([
        ('ss', StandardScaler()),
        ('poly', PolynomialFeatures()),
        ('linear', RidgeCV(alphas=np.logspace(-3, 1, 20)))
    ]),
    Pipeline([
        ('ss', StandardScaler()),
        ('poly', PolynomialFeatures()),
        ('linear', LassoCV(alphas=np.logspace(-3, 1, 20)))
    ])
]
# 参数字典, 字典中的key是属性的名称, value是可选的参数列表
parameters = {
    'poly__degree': [3, 2, 1],
    'poly__interaction_only': [True, False],
    'poly__include_bias': [True, False],
    'linear__fit_intercept': [True, False]
}
# 数据分割
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=0)

titles = ['Ridge', 'Lasso']
colors = ['g-', 'b-']
plt.figure(figsize=(16, 8), facecolor='w')
ln_x_test = range(len(x_test))
plt.plot(ln_x_test, y_test, 'r-', lw=2, label=u'真实值')
for t in range(2):
    print('t', t)
    # 获取模型并设置参数
    # GridSearchCV: 进行交叉验证, 选择出最优的参数值出来
    # 第一个输入参数: 进行参数选择的模型
    # param_grid: 用于进行模型选择的参数字段, 要求是字典类型; cv:进行几折交叉验证
    model = GridSearchCV(models[t], param_grid=parameters, cv=5, n_jobs=1)
    model.fit(x_train, y_train)
    # 模型效果值获取(最优参数)
    print('%s算法:最优参数:' % titles[t], model.best_params_)
    print('%s算法:R2值=%.3f' % (titles[t], model.best_score_))
    y_predict = model.predict(x_test)
    plt.plot(ln_x_test, y_predict, colors[t], lw=t + 3, label=u'%s算法估计值,$R^2$=%.3f' % (titles[t], model.best_score_))
plt.legend(loc='upper left')
plt.grid(True)
plt.title(u'波士顿房屋价格预测')
plt.show()

## 模型训练 ===>单个Lasso模型(一阶特征选择)
model = Pipeline([
    ('ss', StandardScaler()),
    ('poly', PolynomialFeatures(degree=1, include_bias=False, interaction_only=True)),
    ('linear', LassoCV(alphas=np.logspace(-3, 1, 20), fit_intercept=False))
])
model.fit(x_train, y_train)

print('参数: ', list(zip(names, model.get_params()['linear'].coef_)))
print('截距: ', model.get_params()['linear'].intercept_)
