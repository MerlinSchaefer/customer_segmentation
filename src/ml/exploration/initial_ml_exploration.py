# Databricks notebook source
# MAGIC %pip install catboost

# COMMAND ----------

import sys
sys.path.append("..")
import json
import pandas as pd
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.classification import (LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier, MultilayerPerceptronClassifier, LinearSVC)
from pyspark.sql import functions as F
from pyspark.sql import DataFrame, Row
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import OneVsRest
from ml_utils import SklearnModelTester, SparkMLModelTester, CatBoostModelTester
from pyspark.sql.functions import col, udf
from pyspark.ml.linalg import SparseVector, DenseVector
from pyspark.sql.types import StringType
from pyspark.ml.linalg import VectorUDT



# COMMAND ----------

scaled_features = spark.read.table("customer_segmentation_dev.gold_scaled_features")
unscaled_features = spark.read.table("customer_segmentation_dev.gold_no_scaled_features")
target = spark.read.table("customer_segmentation_dev.gold_customer_ml_target")

# COMMAND ----------

def adjust_labels(df, label_col="label"):
    """
    Adjusts labels from 1-4 to 0-3 for compatibility with Spark ML models.
    
    Parameters:
    - df: DataFrame containing the label column.
    - label_col: Name of the label column to be adjusted.

    Returns:
    - DataFrame with labels adjusted to be in the range [0, 3].
    """
    adjusted_df = df.withColumn(label_col, F.when(F.col(label_col) == 1, 0)
                                           .when(F.col(label_col) == 2, 1)
                                           .when(F.col(label_col) == 3, 2)
                                           .when(F.col(label_col) == 4, 3))
    return adjusted_df
target = adjust_labels(target, label_col="segmentation")

# COMMAND ----------

scaled_df = scaled_features.join(target, "id").withColumnRenamed("segmentation", "label").withColumn("label", F.col("label").cast("double"))
unscaled_df = unscaled_features.join(target, "id").withColumnRenamed("segmentation", "label").withColumn("label", F.col("label").cast("double"))

# COMMAND ----------


# UDF to convert sparse vectors to dense vectors
def sparse_to_dense(vector):
    if isinstance(vector, SparseVector):
        return DenseVector(vector.toArray())
    else:
        return vector

# Registering the UDF to convert the vector
sparse_to_dense_udf = udf(sparse_to_dense, VectorUDT())

# Apply the UDF to the features column to convert to dense vectors
def convert_sparse_to_dense(df, features_col="features"):
    return df.withColumn(features_col, sparse_to_dense_udf(df[features_col]))

# COMMAND ----------

df_scaled = convert_sparse_to_dense(scaled_df)
df_unscaled = convert_sparse_to_dense(unscaled_df)

# COMMAND ----------

#df_scaled.display()

# COMMAND ----------

# for sklearn models
# Convert the label and features to Pandas DataFrame
# Extract labels as a list
labels_scaled = df_scaled.select("label").rdd.flatMap(lambda x: x).collect()
labes_unscaled = df_unscaled.select("label").rdd.flatMap(lambda x: x).collect()
# Extract features as a list of lists (where each row is a feature vector)
features_scaled = df_scaled.select("features").rdd \
    .map(lambda row: row.features.toArray()) \
    .collect()
features_unscaled = df_unscaled.select("features").rdd \
    .map(lambda row: row.features.toArray()) \
        .collect()
# Create a Pandas DataFrame from the features and labels
pandas_df_scaled = pd.DataFrame(features_scaled)
pandas_df_scaled['label'] = labels_scaled
pandas_df_unscaled = pd.DataFrame(features_unscaled)
pandas_df_unscaled['label'] = labes_unscaled


y_scaled = pandas_df_scaled['label']
X_scaled = pandas_df_scaled.drop('label', axis=1)
y_unscaled = pandas_df_unscaled['label']
X_unscaled = pandas_df_unscaled.drop('label', axis=1)

# COMMAND ----------

# Display the first few rows of the Pandas DataFrame
#display(pandas_df_unscaled)

# COMMAND ----------

model_params = {
    "scaled": [
                (LogisticRegression, {"maxIter":[100],"regParam": [0.001,0.01, 0.1], "elasticNetParam": [0.0, 1.0]}),
        (MultilayerPerceptronClassifier, {"layers": [[34, 16, 8, 4], [34, 68, 16, 4], [34, 68, 68, 4]], "maxIter": [50, 100, 200], "blockSize": [64, 128, 256]}),

    ],
    "unscaled": [
        (DecisionTreeClassifier, {"maxDepth": [5, 10, 15], "minInstancesPerNode": [1, 5, 10]}),
        (RandomForestClassifier, {"numTrees": [10, 50, 100], "maxDepth": [5, 10, 15], "minInstancesPerNode": [1, 5, 10]}),
    ]
}

# COMMAND ----------

model_tester = SparkMLModelTester(df_scaled=scaled_df, df_unscaled=unscaled_df)

# Run all models
model_tester.evaluate_models(model_params=model_params)



# COMMAND ----------

# Get all results as a list of dictionaries
results = model_tester.get_results()



# Convert hyperparameters to JSON string
results_json = [Row(**{k: json.dumps(v) if isinstance(v, dict) else v for k, v in result.items()}) for result in results]

# Create Spark DataFrame from results
results_df = spark.createDataFrame(results_json)

results_df.display()

# COMMAND ----------

results_df.write.mode("append").saveAsTable("customer_segmentation.initial_ml_tests")

# COMMAND ----------

# MAGIC %md
# MAGIC the neural net and random forest performed best and with the least amount of variance between folds and param sets.
# MAGIC will deep dive into random forest, potentially try gradient boosted forests and further NN architectures. 

# COMMAND ----------

# MAGIC %md
# MAGIC Let's try some other techniques to then go into full hyperparameter optimization with a couple model types.

# COMMAND ----------

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC

# COMMAND ----------

xgboost_params = {
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 6, 12],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.5],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

# COMMAND ----------

catboost_params = {
    'iterations': [100, 200, 500],
    'learning_rate': [0.01, 0.1, 0.2],
    'depth': [4, 6, 10],
    'l2_leaf_reg': [1, 3, 5, 9],
    'random_seed': [13]
}

# COMMAND ----------

svc_params = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['poly', 'rbf', 'sigmoid'],
    'degree': [2, 3, 4],
    'gamma': ['scale', 'auto'],
    'coef0': [0.0, 0.5, 1.0]
}

# COMMAND ----------

sklearn_tester_scaled = SklearnModelTester(X_scaled,y_scaled)
sklearn_tester_unscaled = SklearnModelTester(X_unscaled,y_unscaled)
catboost_tester = CatBoostModelTester(X_unscaled,y_unscaled)
# Create a list of models and their parameter grids
models_scaled = [
    (SVC, svc_params),
]
models_unscaled = [
    (XGBClassifier, xgboost_params),
]
models_catboost = [
    (CatBoostClassifier, catboost_params)
]

# COMMAND ----------

catboost_tester.evaluate_models(models_catboost)

# COMMAND ----------

results = catboost_tester.get_results()
results

# COMMAND ----------

# Round metrics
results = [{k: float(round(v, 5)) if isinstance(v, float) else v for k, v in result.items()} for result in results]
# Convert hyperparameters to JSON string
results_json = [Row(**{k: json.dumps(v) if isinstance(v, dict) else v for k, v in result.items()}) for result in results]

results_json

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, DoubleType

# Define schema based on results_json structure
schema = StructType([
    StructField("model", StringType(), True),
    StructField("hyperparameters", StringType(), True),
    StructField("best_parameters", StringType(), True),
    StructField("f1_score", DoubleType(), True),
    StructField("accuracy", DoubleType(), True),
    StructField("weighted_precision", DoubleType(), True),
    StructField("weighted_recall", DoubleType(), True),


])

# Create Spark DataFrame from results_json with schema
results_df = spark.createDataFrame(results_json, schema)

display(results_df)

# COMMAND ----------

results_df.write.mode("append").saveAsTable("customer_segmentation.initial_ml_tests")

# COMMAND ----------

sklearn_tester_scaled.evaluate_models(models_scaled, cv_type="random", n_iter = 30)

# COMMAND ----------

results = sklearn_tester_scaled.get_results()
results

# COMMAND ----------

# Round metrics
results = [{k: float(round(v, 5)) if isinstance(v, float) else v for k, v in result.items()} for result in results]
# Convert hyperparameters to JSON string
results_json = [Row(**{k: json.dumps(v) if isinstance(v, dict) else v for k, v in result.items()}) for result in results]

results_json

# COMMAND ----------


# Create Spark DataFrame from results_json with schema
results_df = spark.createDataFrame(results_json, schema)

display(results_df)

# COMMAND ----------

results_df.write.mode("append").saveAsTable("customer_segmentation.initial_ml_tests")

# COMMAND ----------

sklearn_tester_unscaled.evaluate_models(models_unscaled, cv_type="random",n_iter = 30)


# COMMAND ----------


results = sklearn_tester_unscaled.get_results()
# Round metrics
results = [{k: float(round(v, 5)) if isinstance(v, float) else v for k, v in result.items()} for result in results]
# Convert hyperparameters to JSON string
results_json = [Row(**{k: json.dumps(v) if isinstance(v, dict) else v for k, v in result.items()}) for result in results]

# Create Spark DataFrame from results
results_df = spark.createDataFrame(results_json, schema)

results_df.display()

# COMMAND ----------

results_df.write.mode("append").saveAsTable("customer_segmentation.initial_ml_tests")

# COMMAND ----------

spark.sql("SELECT * FROM customer_segmentation.initial_ml_tests ORDER BY f1_score DESC").display()

# COMMAND ----------

# MAGIC %md
# MAGIC It looks like the tree based models specifically random forests and Catboost work best.
# MAGIC The Neural Net  and XGB could possibly close the gap if we were to test a lot of options. For now however I will focus on the two model types mentioned above. 
