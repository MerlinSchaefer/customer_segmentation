# Databricks notebook source
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.classification import (LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier, MultilayerPerceptronClassifier, LinearSVC)
from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import OneVsRest
from ..ml_utils import SklearnModelTester, SparkMLModelTester

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

from pyspark.sql.functions import col, udf
from pyspark.ml.linalg import SparseVector, DenseVector
from pyspark.sql.types import StringType
from pyspark.ml.linalg import VectorUDT
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

df_scaled.display()

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

# Convert results to a DataFrame (using pandas for easier viewing)
results_df = spark.createDataFrame(results)
display(results_df)

# COMMAND ----------

results_df.write.table("customer_segmentation.initial_ml_tests")

# COMMAND ----------

# MAGIC %md
# MAGIC the neural net and random forest performed best and with the least amount of variance between folds and param sets.
# MAGIC will deep dive into random forest, potentially try gradient boosted forests and further NN architectures. 

# COMMAND ----------

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
