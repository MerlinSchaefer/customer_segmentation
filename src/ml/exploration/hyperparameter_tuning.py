# Databricks notebook source
import sys
sys.path.append("..")
from pipeline_utils import load_and_prepare_ml_data, convert_to_sklearn_dataframe
from ml_utils import SklearnModelTester, SparkMLModelTester, CatBoostModelTester
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier

# COMMAND ----------

# Create sparkml and sklearn dataframes
df_scaled, df_unscaled = load_and_prepare_ml_data()
X_scaled, y_scaled, X_unscaled, y_unscaled = convert_to_sklearn_dataframe(df_scaled, df_unscaled)
