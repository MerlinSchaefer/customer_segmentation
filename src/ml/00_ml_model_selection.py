# Databricks notebook source
from pipeline_utils import load_and_prepare_ml_data, convert_to_sklearn_dataframe
from ml_utils import SklearnModelTester, SparkMLModelTester, CatBoostModelTester
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier

# COMMAND ----------

# Create sparkml and sklearn dataframes
df_scaled, df_unscaled = load_and_prepare_ml_data()
X_scaled, y_scaled, X_unscaled, y_unscaled = convert_to_sklearn_dataframe(df_scaled, df_unscaled)

# COMMAND ----------

# test pyspark models
model_params = {
    "scaled": [
                (LogisticRegression, {"maxIter":[100],"regParam": [0.001,0.01, 0.1], "elasticNetParam": [0.0, 1.0]}),

    ],
    "unscaled": [
        (RandomForestClassifier, {"numTrees": [10, 50, 100], "maxDepth": [5, 10, 15], "minInstancesPerNode": [1, 5, 10]}),
    ]
}

# COMMAND ----------

model_tester = SparkMLModelTester(df_scaled=df_scaled, df_unscaled=df_unscaled)

# Run all models
model_tester.evaluate_models(model_params=model_params)

# COMMAND ----------

# append to dataframe

# COMMAND ----------

# train sklearn models

# COMMAND ----------

#append to df

# COMMAND ----------

# train catboost

# COMMAND ----------

# append to df

# COMMAND ----------

# find best model in df and refit

# COMMAND ----------

# save best model and register as current model
