# Databricks notebook source
# MAGIC %md #Exploratory Data Analysis 

# COMMAND ----------

# MAGIC %md *** Visualizations for EDA was performed using Tableau, Excel and Python. We have included the ones using Python below.  

# COMMAND ----------

from pyspark.sql.functions import unix_timestamp
from datetime import datetime
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DateType
import pandas as pd
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import col, countDistinct
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %md **Exploring Training Dataset**

# COMMAND ----------

#Creating dataframe for training data
train_df=spark.sql("SELECT * FROM traindf")
train_df.show()

# COMMAND ----------

# MAGIC %md **Viewing total sales(along with scaling the numbers) across Store State and Item Family**

# COMMAND ----------

a = spark.sql("SELECT StoreState, ItemFamily, Log10(SUM(Units)) as sales from traindf GROUP BY StoreState, ItemFamily") 
a.show()
a = a.toPandas()

# COMMAND ----------

# MAGIC %md **Creating a heat map to view sales across store states and item family**

# COMMAND ----------

b = a.dropna().pivot(index = 'ItemFamily', columns = 'StoreState', values = 'sales')
plt.figure(figsize = (10,10))
sns.heatmap(b, linewidths = .5, cmap = "gist_heat_r")
display()

# COMMAND ----------

# MAGIC %md **Viewing table with Item Family and year along with sum of units**

# COMMAND ----------

from pyspark.sql.functions import year, month, dayofmonth
fig1 = train_df.select(year("Date"), "ItemFamily", "units")
fig1.show()
fig1.toPandas()

# COMMAND ----------

# MAGIC %md **Viewing sales of top cities in Ecuador**

# COMMAND ----------

display(stateCity1)

# COMMAND ----------

# MAGIC %md **Viewing count of units sold across all states of Ecuador**

# COMMAND ----------

stateSales = spark.sql("SELECT StoreState, COUNT(Units) FROM traindf GROUP BY StoreState")
stateSales.show()
display(stateSales)

# COMMAND ----------

CitySales = spark.sql("SELECT Count(units), StoreCity FROM traindf GROUP BY StoreCity")
CitySales.show()
display(CitySales)

# COMMAND ----------

# MAGIC %md **Viewing units sold acorss item families for two years**

# COMMAND ----------

from pyspark.sql.functions import year, month, dayofmonth
fig1 = train_df.select(year("Date"), "ItemFamily", "units")
fig1.show()
display(fig1)

# COMMAND ----------

# MAGIC %md **Viewing percentage of sales of perishable items across all store states**

# COMMAND ----------

fig2 = train_df.select("ItemPerishable", "StoreState", "units")
fig2.show()
display(fig2)

# COMMAND ----------

# MAGIC %md **Viewing sales across states and store type**

# COMMAND ----------

fig3 = train_df.select("storeType", "StoreState", "units")
fig3.show()
display(fig3)

# COMMAND ----------

# MAGIC %md **Viewing percentage of sales across different item families**

# COMMAND ----------

unit_per_family = spark.sql("Select Count(units) as count, ItemFamily from traindf group by ItemFamily having count > 150000")
unit_per_family.show()    
display(unit_per_family)

# COMMAND ----------

# MAGIC %md **Counting the number of distinct stores**

# COMMAND ----------

unique_store_id = spark.sql("select COUNT(DISTINCT(StoreNum)) from traindf")
unique_store_id.show()

# COMMAND ----------

# MAGIC %md **Counting the number of distinct item families**

# COMMAND ----------

unique_family = spark.sql("select COUNT(DISTINCT(ItemFamily)) from traindf")
unique_family.show()

# COMMAND ----------

# MAGIC %md **Counting number of distinct store cities**

# COMMAND ----------

unique_city = spark.sql("select COUNT(DISTINCT(Storecity)) from traindf")
unique_city.show()

# COMMAND ----------

# MAGIC %md **Counting number of distinct states with stores**

# COMMAND ----------

unique_state = spark.sql("select COUNT(DISTINCT(StoreState)) from traindf")
unique_state.show()

# COMMAND ----------

# MAGIC %md **Viewing the summary of the entire dataset**

# COMMAND ----------

display(train_df.describe())
