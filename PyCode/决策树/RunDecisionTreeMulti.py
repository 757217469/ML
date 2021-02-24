# coding=utf-8
import numpy as np
from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.evaluation import MulticlassMetrics
import time
import pandas as pd
import matplotlib.pyplot as plt


def convert_float(x):
    return 0 if x == '?' else float(x)


def extract_label(record):
    label = (record[-1])
    return float(label) - 1


def extract_features(record, featureEnd):
    # 提取分类特征字段
    numericalFeatures = [convert_float(field) for field in record[0:featureEnd]]
    return numericalFeatures


def PrepareData(sc):
    print('开始导入数据. . .')
    rawData = sc.textFile('covtype.data/covtype.data')
    print('共计: ' + str(rawData.count()) + '笔')
    lines = rawData.map(lambda x: x.split(','))
    labelpointRDD = lines.map(lambda r: LabeledPoint(extract_label(r), extract_features(r, len(r) - 1)))
    trainData, validationData, testData = labelpointRDD.randomSplit([8, 1, 1])
    return trainData, validationData, testData


def PredictData(labelpointRDD, model):
    for lp in labelpointRDD.take(100):
        predict = model.predict(lp.features)
        label = lp.label
        features = lp.features
        result = ('正确' if (label == predict) else '错误')
        print(' 土地条件: 海拔 : ' + str(features[0]) +
              ' 方位 :' + str(features[1]) +
              ' 斜率 :' + str(features[2]) +
              ' 水源垂直距离 :' + str(features[3]) +
              ' 水源水平距离 :' + str(features[4]) +
              ' 9点时阴影 :' + str(features[5]) +
              '. . .==> 预测 :' + str(predict) +
              ' 实际 :' + str(label) + ' 结果 :' + result)


# overfitting 测试
def evaluateModel(model, validationData):
    score = model.predict(validationData.map(lambda p: p.features))
    scoreAndLabels = score.zip(validationData.map(lambda p: p.label))
    metrics = MulticlassMetrics(scoreAndLabels)
    accuracy = metrics.accuracy
    return accuracy


def trainEvaluateModel(trainData, validationData, impurityParm, maxDepthParm, maxBinsParm):
    startTime = time.time()
    model = DecisionTree.trainClassifier(trainData, numClasses=7, categoricalFeaturesInfo={}, impurity=impurityParm,
                                         maxDepth=maxDepthParm, maxBins=maxBinsParm)
    accuracy = evaluateModel(model, validationData)
    duration = time.time() - startTime
    print('训练评估: 使用参数Impurity=' + str(impurityParm) + ', maxDepth=' + str(maxDepthParm) + ', maxBins=' + str(
        maxBinsParm) + '\n' + '==>所需时间=' + str(duration) + '结果accuracy=' + str(accuracy))
    return accuracy, duration, impurityParm, maxDepthParm, maxBinsParm, model


def showchart(df, evalparm, barData, lineData, yMin, yMax):
    # ax = df[barData].plot(kind='bar', title=evalparm, flgsize=(10, 6), legend=True, fontsize=12)
    ax = df[barData].plot(kind='bar', title=evalparm, legend=True, fontsize=12)
    # flgsize 参数无法使用,  用下面这行代替, 设置宽高
    plt.gcf().set_size_inches(10, 6)
    ax.set_xlabel(evalparm, fontsize=12)
    ax.set_ylim([yMin, yMax])
    ax.set_ylabel(barData, fontsize=12)
    ax2 = ax.twinx()
    ax2.plot(df[lineData].values, linestyle='-', marker='o', linewidth=2.0, color='r')
    plt.show()


def evalParameter(trainData, validationData, evalparm, impurityList, maxDepthList, maxBinsList):
    # 训练评估参数
    metrics = [trainEvaluateModel(trainData, validationData, impurity, maxDepth, maxBins)
               for impurity in impurityList
               for maxDepth in maxDepthList
               for maxBins in maxBinsList]
    # 设置当前评估的参数
    if evalparm == 'impurity':
        IndexList = impurityList[:]
    elif evalparm == 'maxDepth':
        IndexList = maxDepthList[:]
    elif evalparm == 'maxBins':
        IndexList = maxBinsList[:]
    # 转换为pandas DataFrame
    df = pd.DataFrame(metrics, index=IndexList,
                      columns=['accuracy', 'duration', 'impurity', 'maxDepth', 'maxBins', 'model'])
    # 显示图形
    showchart(df, 'impurity', 'accuracy', 'duration', 0.6, 1.0)


def parametersEval(trainData, validationData):
    evalParameter(trainData, validationData, 'impurity',
                  impurityList=['gini', 'entropy'],
                  maxDepthList=[15],
                  maxBinsList=[15])
    # 评估stepSize参数使用
    evalParameter(trainData, validationData, 'maxDepth',
                  impurityList=['gini'],
                  maxDepthList=[3, 5, 10, 15, 20, 25],
                  maxBinsList=[15])
    evalParameter(trainData, validationData, 'maxBins',
                  impurityList=['gini'],
                  maxDepthList=[15],
                  maxBinsList=[3, 5, 10, 50, 100, 200])


def evalAllParameter(trainData, validationData, impurityList, maxDepthList, maxBinsList):
    # for 循环训练评估所有参数组合
    metrics = [trainEvaluateModel(trainData, validationData, impurity, maxDepth, maxBins)
               for impurity in impurityList
               for maxDepth in maxDepthList
               for maxBins in maxBinsList]
    # 找出accuracy最大的参数组合
    Smetrics = sorted(metrics, key=lambda k: k[0], reverse=True)
    bestParameter = Smetrics[0]
    # 显示调校后最佳参数组合
    print('调校后最佳参数 : impurity: ' + str(bestParameter[2]) +
          ', maxDepth : ' + str(bestParameter[3]) +
          ', maxBins : ' + str(bestParameter[4]) +
          '\n, 结果accuracy = ' + str(bestParameter[0]))
    # 返回最佳模型
    return bestParameter[5]


if __name__ == '__main__':
    print('RunDecisionTreeMulti')
    sc = SparkContext()
    # sc._conf.set('spark.executor.memory', '32g').set('spark.driver.memory', '32g').set('spark.driver.maxResultsSize','0')
    # sc._conf.set('spark.driver.maxResultsSize', '3g')
    sc._conf.set('')
    print('==========数据准备阶段=============')
    trainData, validationData, testData = PrepareData(sc)
    trainData.persist()
    validationData.persist()
    testData.persist()
    print('==========训练评估阶段=============')
    # AUC, duration, impurityParm, maxDepthParm, maxBinsParm, model = trainEvaluateModel(trainData, validationData,
    #                                                                                    'entropy', 5, 5)
    print('------所有参数训练评估找出最好的参数组合------')
    # parametersEval(trainData, validationData)
    # model = evalAllParameter(trainData, validationData,
    #                          ['gini', 'entropy'],
    #                          [3, 5, 10, 15, 20, 25],
    #                          [3, 5, 10, 50, 100, 200])
    print(trainEvaluateModel(trainData, validationData, 'entropy', 25, 200))

    print('===========预测数据==============')
    # PredictData(testData, model)
    # print(model.toDebugString())
