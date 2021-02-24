# coding=utf-8
import sys
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from pyspark import SparkContext, SparkConf
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.feature import StandardScaler


def convert_float(x):
    return 0 if x == '?' else float(x)


def extract_label(field):
    label = field[-1]
    return float(label)


def extract_features(field, categoriesMap, featureEnd):
    # 提取分类特征字段
    categoryidx = categoriesMap[field[3]]
    categoryFeatures = np.zeros(len(categoriesMap))
    categoryFeatures[categoryidx] = 1
    # 提取数值字段
    numericalFeatures = [convert_float(field) for field in field[4:featureEnd]]
    # 返回 "分类特征字段" + "数值特征字段"
    return np.concatenate((categoryFeatures, numericalFeatures))


def PrepareData(sc):
    rawDataWithHeader = sc.textFile('train.tsv/train.tsv')
    header = rawDataWithHeader.first()
    rawData = rawDataWithHeader.filter(lambda x: x != header)
    rData = rawData.map(lambda x: x.replace('"', ''))
    lines = rData.map(lambda x: x.split('\t'))
    print('标准化之前')
    categoriesMap = lines.map(lambda fields: fields[3]).distinct().zipWithIndex().collectAsMap()
    labelRDD = lines.map(lambda r: extract_label(r))
    featureRDD = lines.map(lambda r: extract_features(r, categoriesMap, len(r) - 1))
    for _ in featureRDD.first():
        print(str(_) + ',')

    print('标准化之后')
    stdScaler = StandardScaler(withMean=True, withStd=True).fit(featureRDD)
    ScalerFeatureRDD = stdScaler.transform(featureRDD)
    for _ in ScalerFeatureRDD.first():
        print(str(_) + ',')
    labelpoint = labelRDD.zip(ScalerFeatureRDD)
    labelpointRDD = labelpoint.map(lambda r: LabeledPoint(r[0], r[1]))
    trainData, validationData, testData = labelpointRDD.randomSplit([8, 1, 1])
    return trainData, validationData, testData, categoriesMap


def PredictData(sc, model, categoriesMap):
    rawDataWithHeader = sc.textFile('test.tsv/test.tsv')
    header = rawDataWithHeader.first()
    rawData = rawDataWithHeader.filter(lambda x: x != header)
    rData = rawData.map(lambda x: x.replace('"', ''))
    lines = rData.map(lambda x: x.split('\t'))
    dataRDD = lines.map(lambda r: (r[0], extract_features(r, categoriesMap, len(r))))
    DescDict = {
        0: '暂时性网页(ephemeral)',
        1: '长青网页(evergreen)'
    }
    for data in dataRDD.take(50):
        predictResult = model.predict(data[1])
        print('网址: ' + str(data[0]) + '\n' + '==>预测:' + str(predictResult) + '说明:' + DescDict[predictResult] + '\n')


def evaluateModel(model, validationData):
    score = model.predict(validationData.map(lambda p: p.features))
    scoreAndLabels = score.map(lambda _: float(_)).zip(validationData.map(lambda p: p.label))
    metrics = BinaryClassificationMetrics(scoreAndLabels)
    AUC = metrics.areaUnderROC
    return AUC


def trainEvaluateModel(trainData, validationData, numIterations, stepSize, regParam):
    startTime = time.time()
    model = SVMWithSGD.train(trainData, numIterations, stepSize, regParam)
    AUC = evaluateModel(model, validationData)
    duration = time.time() - startTime
    print('训练评估: 使用参数numIterations=' + str(numIterations) + ', stepSize=' + str(stepSize) + ', regParam=' + str(
        regParam) + '\n' + '==>所需时间=' + str(duration) + '结果AUC=' + str(AUC))
    return AUC, duration, numIterations, stepSize, regParam, model


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


def evalParameter(trainData, validationData, evalparm, numIterationsList, stepSizeList, regParamList):
    # 训练评估参数
    metrics = [trainEvaluateModel(trainData, validationData, numIterations, stepSize, regParam)
               for numIterations in numIterationsList
               for stepSize in stepSizeList
               for regParam in regParamList]
    # 设置当前评估的参数
    if evalparm == 'numIterations':
        IndexList = numIterationsList[:]
    elif evalparm == 'stepSize':
        IndexList = stepSizeList[:]
    elif evalparm == 'regParam':
        IndexList = regParamList[:]
    # 转换为pandas DataFrame
    df = pd.DataFrame(metrics, index=IndexList,
                      columns=['AUC', 'duration', 'numIterations', 'stepSize', 'regParam', 'model'])
    # 显示图形
    showchart(df, evalparm, 'AUC', 'duration', 0.5, 0.7)


def parametersEval(trainData, validationData):
    evalParameter(trainData, validationData, 'numIterations',
                  numIterationsList=[1, 3, 5, 15, 25],
                  stepSizeList=[100],
                  regParamList=[1])
    # 评估stepSize参数使用
    evalParameter(trainData, validationData, 'stepSize',
                  numIterationsList=[25],
                  stepSizeList=[10, 50, 100, 200],
                  regParamList=[1])
    evalParameter(trainData, validationData, 'regParam',
                  numIterationsList=[25],
                  stepSizeList=[100],
                  regParamList=[0.01, 0.1, 1])


def evalAllParameter(trainData, validationData, numIterationsList, stepSizeList, regParamList):
    # for 循环训练评估所有参数组合
    metrics = [trainEvaluateModel(trainData, validationData, numIterations, stepSize, regParam)
               for numIterations in numIterationsList
               for stepSize in stepSizeList
               for regParam in regParamList]
    # 找出AUD最大的参数组合
    Smetrics = sorted(metrics, key=lambda k: k[0], reverse=True)
    bestParameter = Smetrics[0]
    # 显示调校后最佳参数组合
    print('调校后最佳参数 : numIterations: ' + str(bestParameter[2]) +
          ', stepSize : ' + str(bestParameter[3]) +
          ', regParam : ' + str(bestParameter[4]) +
          '\n, 结果AUC = ' + str(bestParameter[0]))
    # 返回最佳模型
    return bestParameter[5]


if __name__ == '__main__':
    print('RunSVMWithSGDBinary')
    sc = SparkContext()
    print('==========数据准备阶段=============')
    trainData, validationData, testData, categoriesMap = PrepareData(sc)
    trainData.persist()
    validationData.persist()
    testData.persist()
    print('==========训练评估阶段=============')
    AUC, duration, numIterationsParm, stepSizeParm, regParam, model = trainEvaluateModel(trainData, validationData,
                                                                                         15, 10, .5)
    parametersEval(trainData, validationData)

    print('---所有参数训练评估找出最好的参数组合---')
    model = evalAllParameter(trainData, validationData,
                             [1, 3, 5, 15, 25],
                             [10, 50, 100, 200],
                             [0.01, 0.1, 1])
    auc = evaluateModel(model, testData)
    PredictData(sc, model, categoriesMap)
