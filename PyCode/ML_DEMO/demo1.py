import numpy as np
import os
import struct


def load_mnist(path, kind='train'):
    # 构造文件所在的路径
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)
    return images, labels


images_train_int8, labels_train = load_mnist('D:/PyCode/ML_DEMO/data', kind='train')

images_test_int8, labels_test = load_mnist('D:/PyCode/ML_DEMO/data', kind='t10k')
# 减小阈值, 从255转为2值化
thresh = 50
# 当大于 thresh 时为1, 否则为0   乘 1 是转为整形
images_train = (images_train_int8 >= thresh) * 1
images_test = (images_test_int8 >= thresh) * 1
import matplotlib.pyplot as plt

# 创建整体图像  里面包含10个小图 2行5列
fig, ax = plt.subplots(
    nrows=2,
    ncols=5,
    sharex=True,
    sharey=True,
)
# 图像/轴 展平
ax = ax.flatten()
for i in range(10):
    img = images_train[labels_train == i][0].reshape(28, 28)
    # cmap == color map 灰色    interpolation 插值
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
# 设置横纵坐标轴 , 里面为[] 意思为自适应
ax[0].set_xticks([])
ax[0].set_yticks([])
# 紧凑布局的方式
plt.tight_layout()
plt.show()

# 导入sklearn当中的随机森林模型
from sklearn.ensemble import RandomForestClassifier
import datetime

# 使用sklearn的时候
# 1 指定一个模型
# 2 喂数据, 根据输入数据, 在已选择的模型基础上进行训练
# 3 使用训练好的模型进行预测

# 指定需要使用的模型(还没有涉及到数据)
# n_estimator 指定树的数量   n_jobs=-1 进行并行计算
model = RandomForestClassifier(n_estimators=50, n_jobs=-1)

# 喂数据, 使用训练集的数据进行模型的训练
starttime = datetime.datetime.now()
model.fit(images_train, labels_train)
endtime = datetime.datetime.now()
time_train = endtime - starttime
print(time_train)

# 使用训练好的模型进行预测
model.predict(images_test)  # 传入需要预测的向量/特征

import cv2

# 1. 读取图片
# 2. 转化图片格式
# 3. 预处理
# 4. 向量化
# 5. 进模型
# 使用imread函数读取图像
im = cv2.imread('D:/PyCode/ML_DEMO/data/images/11.jpg')
# 将图像转化为灰度图像
# BGR  opencv中特有的,所有的颜色排列都是从BGR开始的,不像通常使用的是RGB
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# 重置图像大小
im_final = cv2.resize(im_gray, (28, 28))

# 减小阈值
thresh = 30
im_bi = (im_final >= thresh) * 1
# 展平
im_vec = im_bi.flatten()
# 交换横纵  784*1 转为 1*784
im_vec = im_vec.reshape(1, -1)

# 展示图像
cv2.imshow('666', im_final)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 预测
model.predict(im_vec)

# 模型持久化部分
# 导入sklearn中的模型持久化工具
from sklearn.externals import joblib

# 指定一个名称
filename = 'D:/PyCode/ML_DEMO/data/model.m'
# 调用dump函数
joblib.dump(model, filename)
# 调用已经存储的模型
model = joblib.load('D:/PyCode/ML_DEMO/data/model.m')

####################评估#########################

# 预测测试集当中所有数据的结果
# 手动实现 准确率统计
prediction_10000 = model.predict(images_test)
ac = (labels_test == prediction_10000)
count = 0
for i in range(len(ac)):
    if ac[i] == True:
        count = count + 1
accu = count / len(ac)
print(accu)

# 导入统计准确率的方法
from sklearn.metrics import accuracy_score

# 调用工具进行准确率计算
accuracy_sk = accuracy_score(labels_test, prediction_10000)
print(accuracy_sk)

# 调用sklearn进行召回率计算
from sklearn.metrics import recall_score

recall_sk = recall_score(labels_test, prediction_10000, average='macro')
print(recall_sk)

# F1-score
from sklearn.metrics import f1_score

f1_score_sk = f1_score(labels_test, prediction_10000, average='macro')
print(f1_score_sk)

# 生成一个模型评估报告
from sklearn.metrics import classification_report

# 数据集当中包含的label
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# 给每个类别起一个名称
classes = ['class 0',
           'class 1'
           'class 2'
           'class 3'
           'class 4'
           'class 5'
           'class 6'
           'class 7'
           'class 8'
           'class 9'
           ]
# 使用函数生成总体的报告
print(classification_report(labels_test,
                            prediction_10000,
                            target_names=classes,
                            labels=labels,
                            digits=4))  # digits 小数点后保留4位
