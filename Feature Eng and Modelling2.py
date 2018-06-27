# Databricks notebook source
# MAGIC %md #Creating Models using Linear Regression and Random Forest(Store ID grouping)

# COMMAND ----------

# MAGIC %md ***Importing Packages***

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

holiday_events = spark.sql("SELECT * FROM holidays_events_csv")
items = spark.sql("SELECT * FROM items_csv")
stores = spark.sql("SELECT * FROM stores_csv")
transactions = spark.sql("SELECT cast(date as date), store_nbr, transactions FROM transactions_csv")
oil = spark.sql("SELECT cast(date as date), dcoilwtico FROM oil_csv")

# COMMAND ----------

train_df = spark.sql("SELECT * FROM traindf")
#train_df.show()

# COMMAND ----------

train_df = train_df.withColumnRenamed("Date","Date_Date")


# COMMAND ----------

from pyspark.sql.functions import date_format
df3 = train_df.select('Date_Date', date_format('Date_Date', 'u').alias('dow_number'), date_format('Date_Date', 'E').alias('dow_string'))
df3 = df3.distinct()
df3 = df3.withColumnRenamed("Date_Date","Date2")

# COMMAND ----------

store_num_data = train_df.groupBy("StoreNum","ItemFamily", "Date_Date").sum("Units").orderBy("Date_Date")

store_num_data = store_num_data.join(oil,(oil.date == store_num_data.Date_Date), "left").join(stores, (stores.store_nbr == train_df.StoreNum)).join(df3,df3.Date2 == store_num_data.Date_Date,"left")

# COMMAND ----------

#store_num_data = store_num_data.withColumn("OilPrice", store_num_data["dcoilwtico"].cast("float"))

# COMMAND ----------

# MAGIC %md ***Split date into Day, Month, Year***

# COMMAND ----------

split_date=fn.split(store_num_data['Date_Date'], '-') 
store_num_data= store_num_data.withColumn('Year', split_date.getItem(0).cast("int"))
store_num_data= store_num_data.withColumn('Month', split_date.getItem(1).cast("int"))
store_num_data= store_num_data.withColumn('Day', split_date.getItem(2).cast("int"))

store_num_data= store_num_data.withColumn('StoreNum', store_num_data["StoreNum"].cast("int"))
store_num_data = store_num_data.withColumn("Cluster", store_num_data["cluster"].cast("int"))
store_num_data = store_num_data.withColumn("dow_number", store_num_data["dow_number"].cast("int"))
#store_num_data.show(50)

# COMMAND ----------

store_num_data.show(10)

# COMMAND ----------

# MAGIC %md *** Using Indexer and VeryHotEncoder to create vectors***

# COMMAND ----------

indexer = StringIndexer(inputCol="ItemFamily", outputCol="ItemFamilyNum").fit(store_num_data)
store_num_data_indexed = indexer.transform(store_num_data)
#store_num_data_indexed.show()

indexer = StringIndexer(inputCol="type", outputCol="typeNum").fit(store_num_data_indexed)
store_num_data_indexed = indexer.transform(store_num_data_indexed)
#store_num_data_indexed.show()


# COMMAND ----------

encoder = OneHotEncoderEstimator(inputCols=["typeNum","Cluster","StoreNum","ItemFamilyNum","Month","Day","dow_number"],
                                 outputCols=["typeNumVec","ClusterVec","StoreNumVec","ItemFamilyNumVec", "MonthVec","DayVec","DOWNum"])
model = encoder.fit(store_num_data_indexed)
store_num_data_ind_enc = model.transform(store_num_data_indexed)
#store_num_data_ind_enc.show(5)

# COMMAND ----------

store_num_data_ind_enc = store_num_data_ind_enc.withColumnRenamed("store_num_data_ind_enc.Date", "Date_Date")

# COMMAND ----------

# MAGIC %md ***Using Linear Regression to create model***

# COMMAND ----------

trainingdf = store_num_data_ind_enc.filter(store_num_data_ind_enc.Date_Date <'2014-04-28')

# COMMAND ----------

validationdf = store_num_data_ind_enc.filter(store_num_data_ind_enc.Date_Date >= "2014-04-28").filter(store_num_data_ind_enc.Date_Date <= "2014-10-31")

# COMMAND ----------

testdf = store_num_data_ind_enc.filter(store_num_data_ind_enc.Date_Date >= "2014-10-31")

# COMMAND ----------

va = feature.VectorAssembler(inputCols=['typeNumVec','ClusterVec', 'StoreNumVec',"ItemFamilyNumVec","MonthVec","DayVec","DOWNum"], outputCol='features')
lr = regression.LinearRegression(featuresCol='features', labelCol='sum(Units)', regParam=0.5, elasticNetParam=0.3, fitIntercept= True)
pipe = Pipeline(stages=[va, lr])
model = pipe.fit(trainingdf)

# COMMAND ----------

# MAGIC %md *** Calculating RMSE***

# COMMAND ----------

rmse = (fn.avg((fn.col('sum(Units)') - fn.col('prediction'))**2))**.5

# COMMAND ----------

rmse1 = (fn.avg((fn.col('sum(sum(Units))') - fn.col('sum(prediction)'))**2))**.5

# COMMAND ----------

model.transform(validationdf).select(rmse).show()

# COMMAND ----------

model.transform(testdf).select(rmse).show()

# COMMAND ----------

predictedSales= model.transform(testdf)

# COMMAND ----------

predictedSalesByStore = predictedSales.groupBy("Month","Day").sum('sum(Units)', 'prediction')
predictedSalesByStore.show(50)

# COMMAND ----------

 abc = predictedSalesByStore.filter(predictedSalesByStore.Month == 1)

# COMMAND ----------

abc = abc.withColumnRenamed("sum(sum(Units))","ActualSales").withColumnRenamed("sum(prediction)","PredictedSales")

# COMMAND ----------

abc.write.saveAsTable("PredictedSalesForJan2015")

# COMMAND ----------

predictedSalesByStore.withColumn("sum(prediction)", predictedSalesByStore["sum(prediction)"].cast("bigint")).show()

# COMMAND ----------

predictedSalesByStore.select(rmse1).show()

# COMMAND ----------

# MAGIC %md *** Using Random Forest Modelling***

# COMMAND ----------

va1 = feature.VectorAssembler(inputCols=['typeNumVec','ClusterVec', 'StoreNumVec',"ItemFamilyNumVec","MonthVec","DayVec", "DOWNum"], outputCol='features')
lr1 = regression.RandomForestRegressor(featuresCol='features', labelCol='sum(Units)')
pipe1 = Pipeline(stages=[va1, lr1])
model1 = pipe1.fit(trainingdf)

# COMMAND ----------

model1.transform(validationdf).select(rmse).show()

# COMMAND ----------

model1.transform(testdf).select(rmse).show()

# COMMAND ----------

model.stages[1].coefficients

# COMMAND ----------

tester = store_num_data

# COMMAND ----------

from pyspark.sql.functions import date_format
df3 = tester.select('Date_Date', date_format('Date_Date', 'u').alias('dow_number'), date_format('Date_Date', 'E').alias('dow_string'))
df3 = df3.distinct()

