# coding=utf-8
import numpy as np

np.random.seed(0)


# 计算损失函数
def compute_cost(theta, x, y):
    cost = 0
    m = len(x)
    for i in range(m):
        cost += (y[i] - np.dot(theta, x[i])) ** 2
    return cost / (2 * m)


# 计算梯度
def step_grad_desc(theta, x, y, alpha):
    m = len(x)
    # i = np.random.randint(0, m - 1)
    grad_theta = np.zeros((1, len(x[0])))
    for i in range(m):
        grad_theta += (np.dot(theta, x[i]) - y[i]) * x[i]
    theta -= alpha * grad_theta / m
    print('theta: ', theta)
    return theta


# GD
def grad_desc(x, y, alpha=[0.01], iter_num=20000):
    # x的第0位对应intercept_
    x = np.array([[1] + i for i in x.tolist()])
    # initial theta
    last_cost = 0
    theta = np.zeros((1, len(x[0])))
    for a in alpha:
        for i in range(iter_num):
            print(i)
            theta = step_grad_desc(theta, x, y, a)
            cost = compute_cost(theta, x, y)
            print('cost: %.20f' % cost)
            if last_cost == cost[0]:
                break
            else:
                last_cost = cost[0]
    return theta


# 生成数据
N = 10
x = np.linspace(0, 6, N) + np.random.randn(N)
y = 1.8 * x ** 3 + x ** 2 - 14 * x - 7 + np.random.randn(N)
x.shape = -1, 1
y.shape = -1, 1
theta = grad_desc(x, y)
