# Databricks notebook source
# MAGIC %md #Creasting models using Linear Regression and Random Forest ( State wise groups)

# COMMAND ----------

from __future__ import division, absolute_import
from pyspark.sql import Row
from pyspark.ml import regression
from pyspark.ml import feature
from pyspark.ml import Pipeline
from pyspark.sql import functions as fn
from pyspark.ml.feature import OneHotEncoderEstimator
from pyspark.ml.feature import StringIndexer

# COMMAND ----------

items = spark.sql("SELECT cast(item_nbr as int) as item_Num, family, cast(class as int) as class, cast(perishable as int) as perishable FROM items_csv")
stores = spark.sql("SELECT cast(store_nbr as int) as store_Num,city,state,type, cast(cluster as int) as cluster FROM stores_csv")

# COMMAND ----------

train_df = spark.sql("SELECT * FROM traindf where StoreState ='Pichincha'")

# COMMAND ----------

train_df.show()

# COMMAND ----------

train_df = train_df.withColumnRenamed("Date","Date_Date")

# COMMAND ----------

from pyspark.sql.functions import date_format
df3 = train_df.select('Date_Date', date_format('Date_Date', 'u').alias('dow_number'), date_format('Date_Date', 'E').alias('dow_string'))
df3 = df3.distinct()
df3 = df3.withColumnRenamed("Date_Date","Date2")

# COMMAND ----------

store_dept_data = train_df.groupBy("StoreState","ItemFamily", "Date_Date").sum("Units").orderBy("Date_Date").join(df3,df3.Date2 == train_df.Date_Date)

# COMMAND ----------

store_dept_data.show(10)

# COMMAND ----------

store_dept_data = store_dept_data.join(stores, (stores.state == store_dept_data.StoreState), "left")

# COMMAND ----------

# MAGIC %md ***Split Day, Month, Year from the given date***

# COMMAND ----------

split_date=fn.split(store_dept_data['Date_Date'], '-') 
store_dept_data= store_dept_data.withColumn('Year', split_date.getItem(0).cast("int"))
store_dept_data= store_dept_data.withColumn('Month', split_date.getItem(1).cast("int"))
store_dept_data= store_dept_data.withColumn('Day', split_date.getItem(2).cast("int"))
#store_dept_data.show(50)
store_dept_data= store_dept_data.withColumn('dow_number', store_dept_data.dow_number.cast("int"))

# COMMAND ----------

# MAGIC %md *** Using indexer and VeryHotEncoder to create vectors***

# COMMAND ----------

indexer = StringIndexer(inputCol="ItemFamily", outputCol="ItemFamilyNum").fit(store_dept_data)
store_dept_data_indexed = indexer.transform(store_dept_data)

indexer3 = StringIndexer(inputCol="cluster", outputCol="ClusterNum").fit(store_dept_data_indexed)
store_dept_data_indexed = indexer3.transform(store_dept_data_indexed)

indexer4 = StringIndexer(inputCol="type", outputCol="TypeNum").fit(store_dept_data_indexed)
store_dept_data_indexed = indexer4.transform(store_dept_data_indexed)


# COMMAND ----------

store_dept_data_indexed= store_dept_data_indexed.withColumn('dow_number', store_dept_data_indexed.dow_number.cast("int"))

# COMMAND ----------

encoder = OneHotEncoderEstimator(inputCols=["Month","ClusterNum","TypeNum", "Day","dow_number"],
                                 outputCols=["MonthVec","ClusterNumVec","TypeNumVec","DayVec", "DOWNumVec"])
model = encoder.fit(store_dept_data_indexed)
store_dept_data_ind_enc = model.transform(store_dept_data_indexed)


# COMMAND ----------

# MAGIC %md *** Linear Regression modelling***

# COMMAND ----------

trainingdf = store_dept_data_ind_enc.filter(store_dept_data_ind_enc.Date_Date <'2014-04-28')

# COMMAND ----------

validationdf = store_dept_data_ind_enc.filter(store_dept_data_ind_enc.Date_Date >= "2014-04-28").filter(store_dept_data_ind_enc.Date_Date <= "2014-11-04")

# COMMAND ----------

testdf = store_dept_data_ind_enc.filter(store_dept_data_ind_enc.Date_Date >= "2014-11-04")

# COMMAND ----------

va = feature.VectorAssembler(inputCols=["MonthVec","ClusterNumVec", "TypeNumVec","DayVec", "DOWNumVec"], outputCol='features')
lr = regression.LinearRegression(featuresCol='features', labelCol='sum(Units)', regParam=20, elasticNetParam=0.9)
pipe = Pipeline(stages=[va, lr])
model = pipe.fit(trainingdf)

# COMMAND ----------

model.transform(validationdf).select(rmse).show()

# COMMAND ----------

rmse = fn.avg((fn.col('sum(Units)') - fn.col('prediction'))**2))

# COMMAND ----------

rmse1 = (fn.avg((fn.col('sum(sum(Units))') - fn.col('sum(prediction)'))**2))**.5

# COMMAND ----------

predictedSalesByState = model.transform(testdf)

# COMMAND ----------

predictedSalesByState = predictedSalesByState.groupBy("ItemFamily","Month","Day").sum('sum(Units)', 'prediction')
#predictedSalesByStore.show(50)

# COMMAND ----------

abc = predictedSalesByState.filter(predictedSalesByState.Month == 1)


# COMMAND ----------

abc = abc.withColumnRenamed("sum(sum(Units))","ActualSales").withColumnRenamed("sum(prediction)","PredictedSales")
abc.show()
#abc.write.saveAsTable("predictedSalesByItemFam1")

# COMMAND ----------

model.stages[1].coefficients

# COMMAND ----------

# MAGIC %md *** Random Forest Modelling***

# COMMAND ----------

va1 = feature.VectorAssembler(inputCols=["MonthVec","DayVec", "DOWNumVec"], outputCol='features')
lr1 = regression.RandomForestRegressor(featuresCol='features', labelCol='sum(Units)')
pipe = Pipeline(stages=[va1, lr1])
model1 = pipe.fit(trainingdf)

# COMMAND ----------

model.transform(validationdf).select(rmse).show()
