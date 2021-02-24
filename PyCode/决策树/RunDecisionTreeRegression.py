# coding=utf-8
import numpy as np
from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.evaluation import RegressionMetrics
import time
import pandas as pd
import matplotlib.pyplot as plt
import math


def convert_float(x):
    return 0 if x == '?' else float(x)


def extract_label(record):
    label = (record[-1])
    return float(label)


def extract_features(record, featureEnd):
    # 提取分类特征字段
    featureSeason = [convert_float(field) for field in record[2]]
    features = [convert_float(field) for field in record[4: featureEnd - 2]]
    return np.concatenate((featureSeason, features))


def PrepareData(sc):
    print('开始导入数据. . .')
    rawDataWithHeader = sc.textFile('Bike-Sharing-Dataset/hour.csv')
    header = rawDataWithHeader.first()
    rawData = rawDataWithHeader.filter(lambda x: x != header)
    lines = rawData.map(lambda x: x.split(','))
    print(lines.first())
    print('共计: ' + str(lines.count()) + ' 项')
    labelpointRDD = lines.map(lambda r: LabeledPoint(extract_label(r), extract_features(r, len(r) - 1)))
    print(labelpointRDD.first())
    trainData, validationData, testData = labelpointRDD.randomSplit([8, 1, 1])
    print(' 将数据分 trainData:' + str(validationData.count()) +
          'validationData' + str(trainData.count) +
          'testData:' + str(testData.count())
          )
    return trainData, validationData, testData


def PredictData(labelpointRDD, model):
    SeasonDict = {1: '春', 2: '夏', 3: '秋', 4: '冬'}
    HolidayDict = {0: ' 非假日 ', 1: ' 假日 '}
    WeekDict = {0: ' 一 ', 1: ' 二 ', 2: ' 三 ', 3: ' 四 ', 4: ' 五 ', 5: ' 六 ', 6: ' 日 '}
    WorkDayDict = {1: ' 工作日 ', 0: ' 非工作日 '}
    WeatherDict = {1: ' 晴 ', 2: ' 阴 ', 3: ' 小雨 ', 4: ' 大雨 '}
    # ------------4.进行预测并显示结果--------------
    for lp in labelpointRDD.take(20):
        predict = int(model.predict(lp.features))
        label = lp.label
        features = lp.features
        result = ' 正确 ' if label == predict else '错误'
        error = math.fabs(label - predict)
        dataDesc = ' 特征 : ' + SeasonDict[features[0]] + ' 季 ,' + \
                   str(features[1]) + ' 月 ,' + str(features[2]) + ' 时 ,' + \
                   HolidayDict[features[3]] + ',' + ' 星期 ' + WeekDict[features[4]] + ',' + \
                   WorkDayDict[features[5]] + ',' + WeatherDict[features[6]] + ',' + \
                   str(features[7] * 41) + '度 ,' + ' 体感' + str(features[8] * 50) + ' 度,' + \
                   ' 湿度' + str(features[9] * 100) + ',' + ' 风速' + str(features[10] * 67) + \
                   ' ==> 预测结果 :' + str(predict) + \
                   ' , 实际 :' + str(label) + result + ', 误差 :' + str(error)
        print(dataDesc)


# overfitting 测试
def evaluateModel(model, validationData):
    score = model.predict(validationData.map(lambda p: p.features))
    scoreAndLabels = score.zip(validationData.map(lambda p: p.label))
    metrics = RegressionMetrics(scoreAndLabels)
    RMSE = metrics.rootMeanSquaredError
    return RMSE


def trainEvaluateModel(trainData, validationData, impurityParm, maxDepthParm, maxBinsParm):
    startTime = time.time()
    model = DecisionTree.trainRegressor(trainData, categoricalFeaturesInfo={}, impurity=impurityParm,
                                        maxDepth=maxDepthParm, maxBins=maxBinsParm)
    RMSE = evaluateModel(model, validationData)
    duration = time.time() - startTime
    print(' 训练评估: 使用参数 ' +
          'impurityParm = %s, ' % impurityParm +
          'maxDepthParm = %s, ' % maxDepthParm +
          'maxBinsParm = %s, ' % maxBinsParm +
          '耗时 %f' % duration,
          '结果RMSE = %f,' % RMSE
          )
    return (RMSE, duration, impurityParm, maxDepthParm, maxBinsParm, model)


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
                      columns=['RMSE', 'duration', 'impurity', 'maxDepth', 'maxBins', 'model'])
    # 显示图形
    showchart(df, 'impurity', 'RMSE', 'duration', 0, 200)


def parametersEval(training_RDD, validation_RDD):
    print('-----评估maxDepth参数使用-----')
    evalParameter(training_RDD, validation_RDD, 'maxDepth',
                  impurityList=['variance'],
                  maxDepthList=[3, 5, 10, 15, 20, 25],
                  maxBinsList=[10])
    print('-----评估maxBins参数使用-----')
    evalParameter(training_RDD, validation_RDD, 'maxBins',
                  impurityList=['variance'],
                  maxDepthList=[10],
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
    print('RunDecisionTreeRegression')
    sc = SparkContext()
    # sc._conf.set('spark.executor.memory', '32g').set('spark.driver.memory', '32g').set('spark.driver.maxResultsSize','0')
    sc._conf.set('spark.driver.maxResultsSize', '0')
    print('==========数据准备阶段=============')
    trainData, validationData, testData = PrepareData(sc)
    trainData.persist()
    validationData.persist()
    testData.persist()
    print('==========训练评估阶段=============')
    # AUC, duration, impurityParm, maxDepthParm, maxBinsParm, model = trainEvaluateModel(trainData, validationData,
    #                                                                                    'entropy', 5, 5)
    print('------所有参数训练评估找出最好的参数组合------')
    parametersEval(trainData, validationData)
    model = evalAllParameter(trainData, validationData,
                             ['variance'],
                             [3, 5, 10, 15, 20, 25],
                             [3, 5, 10, 50, 100, 200])

    print('===========预测数据==============')
    PredictData(testData, model)
    print(model.toDebugString())
