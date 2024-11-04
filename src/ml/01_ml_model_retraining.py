# Databricks notebook source
from pipeline_utils import load_and_prepare_ml_data, convert_to_sklearn_dataframe

# COMMAND ----------

# Create sparkml and sklearn dataframes
df_scaled, df_unscaled = load_and_prepare_ml_data()
X_scaled, y_scaled, X_unscaled, y_unscaled = convert_to_sklearn_dataframe(df_scaled, df_unscaled)

# COMMAND ----------

# get current best model configuration


# COMMAND ----------

# retrain model with new data
