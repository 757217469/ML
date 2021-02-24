# coding=utf-8
from sklearn import datasets
from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

iris = datasets.load_iris()
X, y = iris.data, iris.target
print('样本数量:%d, 特征数量:%d' % X.shape)

# 模型对象构建
# code_size : 置顶最终使用多少个子模型, 实际的子模型的数量=code_size*label_number
clf = OutputCodeClassifier(LinearSVC(random_state=0), code_size=30, random_state=0)
clf.fit(X, y)
# 输出预测结果值
print(clf.predict(X))
print('准确率:%.3f' % accuracy_score(y, clf.predict(X)))
print(clf.score(X, y))
# 模型属性输出
k = 1
for item in clf.estimators_:
    print('第%d个模型:' % k, end='')
    print(item)
    k += 1

print(clf.classes_)
