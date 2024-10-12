# Databricks notebook source
from pyspark.sql.functions import current_timestamp

# COMMAND ----------

spark.sql("CREATE SCHEMA IF NOT EXISTS customer_segmentation")

# COMMAND ----------

data_path = "dbfs:/FileStore/customer_segmentation/original/"

initial_train = spark.read.csv(data_path + "Train.csv", header=True, inferSchema=True)
initial_test = spark.read.csv(data_path + "Test.csv", header=True, inferSchema=True)

# COMMAND ----------

initial_train.display()

# COMMAND ----------

initial_test.display()


# COMMAND ----------

# get on timestamp for all initial insertions
insert_timestamp = current_timestamp()

# COMMAND ----------

(initial_train
             .withColumn("inserted_at", insert_timestamp)
                .write.mode('overwrite')
                .saveAsTable('customer_segmentation.train_raw'))
(initial_test.drop('Segmentation')
             .withColumn("inserted_at", insert_timestamp)
                .write.mode('overwrite')
                .saveAsTable('customer_segmentation.test_raw'))
