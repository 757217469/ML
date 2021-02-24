import pyspark
import numpy as np
from pyspark.sql import Row, SparkSession
from pyspark.sql.functions import udf, col
import pyspark.sql.types
import pandas as pd
import matplotlib.pyplot as plt

sc = pyspark.SparkContext()
sqlContext = SparkSession.builder.getOrCreate()
row_df = sqlContext.read.format('csv').option('header', 'true').option('delimiter', '\t').load(
    'D:/PyCode/spark统计/pythonsparkexample/PythonProject/data/train.tsv')
print(row_df.count())
row_df.printSchema()
row_df.select('url', 'alchemy_category', 'alchemy_category_score', 'is_news', 'label').show(10)


def replace_question(x):
    return '0' if x == '?' else x


replace_question = udf(replace_question)
df = row_df.select(['url', 'alchemy_category'] +
                   [replace_question(col(column)).cast('double').alias(column)
                    for column in row_df.columns[4:]])
df.select('url', 'alchemy_category', 'alchemy_category_score', 'is_news', 'label').show(10)
train_df, test_df = df.randomSplit([0.7, 0.3])
train_df.cache()
test_df.cache()
