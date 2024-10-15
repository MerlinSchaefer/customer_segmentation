# Databricks notebook source
import dlt
from pyspark.sql.functions import col

# COMMAND ----------

# Silver Table: Cleaned and transformed data
@dlt.table(
    comment = "silver table with valid customer data for training"
)
@dlt.expect_all_or_drop(
    {"valid_id": "id IS NOT NULL",  # Ensure ID is not null
    "valid_segmentation": "segmentation IS NOT NULL AND segmentation IN ('A', 'B', 'C', 'D')",  # Ensure Segmentation (target) is not null
    "non_negative_age": "age >= 0",
    "valid_age": "age <= 120",
    "valid_family_size": "family_size >= 0 AND family_size <= 15",
    "valid_work_experience": "work_experience >= 0 AND work_experience <= 50"}
    )
def silver_training_customer_data():
    bronze_df = dlt.read("bronze_training_customer_data")
    # Standardize column names (lowercase, underscores instead of spaces)
    df = bronze_df.toDF(*[col.lower().replace(' ', '_') for col in bronze_df.columns])

    # Convert Age and Family Size to integers, Work Experience to float
    df = (df.withColumn("age", col("age").cast("int"))
          .withColumn("family_size", col("family_size").cast("int"))
          .withColumn("work_experience", col("work_experience").cast("float"))
         )
    return df

