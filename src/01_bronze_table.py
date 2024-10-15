# Databricks notebook source
import dlt
from pyspark.sql.functions import col, when, current_timestamp
from dlt_utils import load_schema_from_yaml

# COMMAND ----------

# Retrieve settings from the Spark configuration (set in the YAML)
schema_path = spark.conf.get("bronze_schema_path")
bronze_schema_name = spark.conf.get("bronze_schema_name")
input_path = spark.conf.get("cloudFiles.inputPath")
file_format = spark.conf.get("cloudFiles.format")
max_files_per_trigger = spark.conf.get("cloudFiles.maxFilesPerTrigger")
header_option = spark.conf.get("cloudFiles.header")

# COMMAND ----------

raw_data_schema = load_schema_from_yaml(schema_path, bronze_schema_name)

# COMMAND ----------

# Bronze Table: Raw data ingestion from DBFS using Auto Loader

@dlt.table
def bronze_training_customer_data():
    return (
        spark.readStream.format("cloudFiles")  # Use Auto Loader
        .option("cloudFiles.format", file_format)
        .option("cloudFiles.header", header_option)  # Ensure header is recognized
        .option("cloudFiles.maxFilesPerTrigger", max_files_per_trigger)
        .schema(raw_data_schema)
        .load(input_path)  # Path and other options are configured in the YAML
        .withColumn("ingest_timestamp", current_timestamp())  # Add ingest timestamp
    )
