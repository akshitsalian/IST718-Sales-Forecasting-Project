# Databricks notebook source
holiday_events = spark.sql("SELECT * FROM holidays_events_csv")
items = spark.sql("SELECT * FROM items_csv")
stores = spark.sql("SELECT * FROM stores_csv")
transactions = spark.sql("SELECT cast(date as date), store_nbr, transactions FROM transactions_csv")

# COMMAND ----------

from pyspark.sql.functions import unix_timestamp
from datetime import datetime
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DateType
import pandas as pd
from pyspark.ml.feature import StringIndexer
import seaborn as sn

train = spark.sql("SELECT id, date, store_nbr,item_nbr, unit_sales,onpromotion FROM traina_csv")
func =  udf (lambda x: datetime.strptime(x, '%m/%d/%Y'), DateType())

train = train.withColumn('date_date', func(col('date')))
train.show(50)


# COMMAND ----------

holiday_events.show(5)

# COMMAND ----------

train.show()

# COMMAND ----------

train_joint = train.join(items,on = "item_nbr", how = "left").join(stores,on="store_nbr", how = "left")

train_joint = train_joint.join(transactions, (transactions.store_nbr == train_joint.store_nbr) & (transactions.date == train_joint.date_date), how = "left")

train_joint.show(5)


# COMMAND ----------

# MAGIC %md Joining Local holidays on dates

# COMMAND ----------

#local_holiday_events = spark.sql("SELECT cast(date as date) as Local_Date, type as Local_type, locale as Local_locale, locale_name as Location FROM holidays_events_csv where locale = 'Local'")
#local_holiday_events.show(50)
#train_joint = train_joint.join(local_holiday_events, ((local_holiday_events.Local_Date == train_joint.date_date) & (local_holiday_events.Location == #train_joint.city)))
#train_joint.show(50)

# COMMAND ----------

train_joint.show(50)

# COMMAND ----------

from pyspark.sql import functions as fn

train_df = train_joint.selectExpr("id as Tid","date_date as Date","item_nbr as ItemNum","traina_csv.store_nbr as StoreNum","unit_sales as Units","onpromotion as OnPromotion","family as ItemFamily","class as ItemClass","perishable as ItemPerishable","city as StoreCity","state as StoreState","type as StoreType","cluster as StoreCluster", "transactions as StoreTransactions")

# COMMAND ----------

train_df=spark.sql("SELECT * FROM traindf")
unit_per_family = spark.sql("Select Count(units) as count, ItemFamily from traindf group by ItemFamily having count > 150000")
unit_per_family.show()    

# COMMAND ----------

display(unit_per_family)

# COMMAND ----------

unique_store_id = spark.sql("select COUNT(DISTINCT(StoreNum)) from traindf")
unique_store_id.show()

# COMMAND ----------

unique_family = spark.sql("select COUNT(DISTINCT(ItemFamily)) from traindf")
unique_family.show()

# COMMAND ----------

unique_city = spark.sql("select COUNT(DISTINCT(Storecity)) from traindf")
unique_city.show()

# COMMAND ----------

unique_state = spark.sql("select COUNT(DISTINCT(StoreState)) from traindf")
unique_state.show()

# COMMAND ----------

display(train_df.describe())

# COMMAND ----------

train_DF=spark.sql("SELECT * from traindf")

# COMMAND ----------

train_Df=train_DF.toPandas()

# COMMAND ----------

display(date_units)
