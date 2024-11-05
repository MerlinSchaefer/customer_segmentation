# Databricks notebook source
from create_data import DataSynthesizer
from datetime import datetime

# COMMAND ----------

today = datetime.today().strftime("%Y-%m-%d")

# COMMAND ----------

input_data_path =  spark.conf.get("input_data_path") #"/dbfs/FileStore/customer_segmentation/train/Train.csv"
train_output_path = spark.conf.get("train_output_path") #"/dbfs/FileStore/customer_segmentation/train/"
test_output_path =  spark.conf.get("test_output_path") #"/dbfs/FileStore/customer_segmentation/test/"

# COMMAND ----------


synthesizer = DataSynthesizer(input_data_path)
print(synthesizer.data.info())
synthesizer.detect_metadata()

# COMMAND ----------

print("creating test data")
test_features, test_target = synthesizer.create_test_data()


# COMMAND ----------

print("creating training data")
train_data = synthesizer.create_train_data()
# TODO: create save and load for synthesizer

# COMMAND ----------


test_features.to_csv(f"{test_output_path}{today}_Test_features.csv", index=False)
test_target.to_csv(f"{test_output_path}{today}_Test_target.csv", index=False)
train_data.to_csv(f"{train_output_path}{today}_Train.csv", index=False)

