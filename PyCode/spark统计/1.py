import pyspark
import numpy as np
from pyspark.sql import Row, SparkSession
import pandas as pd
import matplotlib.pyplot as plt

sc = pyspark.SparkContext()
RawUserRDD = sc.textFile('pythonsparkexample/PythonProject/data/u.user')
userRDD = RawUserRDD.map(lambda x: x.split('|'))
userRDD.take(5)
user_Rows = userRDD.map(lambda p:
                        Row(
                            userId=int(p[0]),
                            age=int(p[1]),
                            gender=p[2],
                            occupation=p[3],
                            zipcode=p[4]
                        ))
user_Rows.take(5)
sqlContext = SparkSession.builder.getOrCreate()
user_df = sqlContext.createDataFrame(user_Rows)
user_df.printSchema()
user_df.show(5)
# 创建别名
df = user_df.alias('df')
# 登录临时表
user_df.registerTempTable('user_table')
sqlContext.sql('select count(1) from user_table').show()
sqlContext.sql('select * from user_table').show()
sqlContext.sql('select * from user_table limit 5').show()
userRDDnew = userRDD.map(lambda x: (x[0], x[3], x[2], x[1]))
userRDDnew.take(5)
user_df.select('userId', 'occupation', 'gender', 'age').show(5)
user_df.select(user_df.userId, user_df.occupation, user_df.gender, user_df.age).show(5)
df.select(df.userId, df.occupation, df.gender, df.age).show(5)
df.select(df['userId'], df['occupation'], df['gender'], df['age']).show(5)
sqlContext.sql('select userId,occupation,gender,age from user_table').show(5)
# 增加计算字段
userRDDnew = userRDD.map(lambda x: (x[0], x[3], x[2], x[1], 2016 - int(x[1])))
userRDDnew.take(5)
df.select('userId', 'occupation', 'gender', 'age', 2016 - df.age).show(5)
df.select('userId', 'occupation', 'gender', 'age', (2016 - df.age).alias('birthyear')).show(5)
sqlContext.sql('select userId,occupation,gender,age,2016-age birthyear from user_table').show(5)
# 过滤
user_df.filter("occupation='technician'").filter("gender='M'").filter("age=24").show()
df.filter((df.occupation == 'technician') & (df.gender == 'M') & (df.age == 24)).show()
df.filter((df['occupation'] == 'technician') & (df['gender'] == 'M') & (df['age'] == 24)).show()
sqlContext.sql('select * from user_table where occupation=="technician" and gender="M" and age=24').show(5)
# 排序
userRDD.takeOrdered(5, key=lambda x: int(x[1]))
userRDD.takeOrdered(5, key=lambda x: -int(x[1]))
sqlContext.sql('select userId,occupation,gender,age from user_table order by age').show(5)
sqlContext.sql('select userId,occupation,gender,age from user_table order by age DESC').show(5)
user_df.select('userId', 'occupation', 'gender', 'age').orderBy('age').show(5)
df.select('userId', 'occupation', 'gender', 'age').orderBy(df.age, ascending=0).show(5)
df.select('userId', 'occupation', 'gender', 'age').orderBy(df.age.desc()).show(5)
userRDD.takeOrdered(5, key=lambda x: (-int(x[1]), x[2]))
sqlContext.sql('select userId,age,gender,occupation,zipcode from user_table order by age desc,gender').show(5)
df.orderBy(['age', 'gender'], ascending=[0, 1]).show(5)
# 去重
userRDD.map(lambda x: x[2]).distinct().collect()
userRDD.map(lambda x: (x[1], x[2])).distinct().take(20)
sqlContext.sql('select distinct gender from user_table').show()
sqlContext.sql('select distinct age,gender from user_table').show()
user_df.select('gender').distinct().show()
user_df.select('age', 'gender').distinct().show()
# 分组统计数据
userRDD.map(lambda x: (x[2], 1)).reduceByKey(lambda x, y: x + y).collect()
userRDD.map(lambda x: ((x[2], x[3]), 1)).reduceByKey(lambda x, y: x + y).collect()
sqlContext.sql('select gender,count(*) counts from user_table group by gender').show()
sqlContext.sql('select gender,occupation,count(*) counts from user_table group by gender,occupation').show(100)
user_df.select('gender').groupby('gender').count().show()
user_df.select('gender', 'occupation').groupby('gender', 'occupation').count().orderBy('gender', 'occupation').show(100)

user_df.stat.crosstab('occupation', 'gender').show(30)
# join 连接数据
rawDataWithHeader = sc.textFile('pythonsparkexample/PythonProject/data/free-zipcode-database-Primary.csv')
rawDataWithHeader.take(2)
header = rawDataWithHeader.first()
rawData = rawDataWithHeader.filter(lambda x: x != header)
rawData.first()
rData = rawData.map(lambda x: x.replace('"', ''))
ZipRDD = rData.map(lambda x: x.split(','))
ZipRDD.first()
zipcode_data = ZipRDD.map(lambda p:
                          Row(
                              zipcode=int(p[0]),
                              zipCodeType=p[1],
                              city=p[2],
                              state=p[3]
                          ))
zipcode_data.take(5)
zipcode_df = sqlContext.createDataFrame(zipcode_data)
zipcode_df.printSchema()
zipcode_df.registerTempTable('zipcode_table')
zipcode_df.show(10)
sqlContext.sql(
    'select u.*,z.city,z.state from user_table u left join zipcode_table z on u.zipcode=z.zipcode where z.state="NY"').show(
    10)
sqlContext.sql(
    'select z.state,count(*) from user_table u left join zipcode_table z on u.zipcode=z.zipcode group by z.state').show(
    60)
joined_df = user_df.join(zipcode_df, user_df.zipcode == zipcode_df.zipcode, 'left_outer')
joined_df.printSchema()
joined_df.show(10)
joined_df.filter('state="NY"').show(10)
GroupByState_df = joined_df.groupBy('state').count()
GroupByState_df.show(60)
GroupByState_pandas_df = GroupByState_df.toPandas().set_index('state')
# %matplotlib inline   qupyter时使用,在console面板输出
# ax = GroupByState_pandas_df['count'].plot(kind='bar',title='State',figsize=(12,6),legend=True,fontsize=12)
ax = GroupByState_pandas_df['count'].plot(kind='bar', title='State', legend=True, fontsize=12)
plt.gcf().set_size_inches(12, 6)
plt.show()
Occupation_df = sqlContext.sql('select u.occupation,count(*) counts from user_table u group by occupation')
Occupation_df.show(30)
Occupation_pandas_df = Occupation_df.toPandas().set_index('occupation')
print(Occupation_pandas_df)
ax = Occupation_pandas_df['counts'].plot(kind='pie',
                                         title='occupation', figsize=(8, 8),
                                         startangle=90, autopct='%1.1f%%')
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
