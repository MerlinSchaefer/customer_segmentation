# Databricks notebook source
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.classification import (LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier, MultilayerPerceptronClassifier, LinearSVC)
from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import OneVsRest

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

# test LR
lr = LogisticRegression(maxIter=100, regParam=0.1,featuresCol="features", labelCol="label")

# COMMAND ----------

lr_model = lr.fit(scaled_df)

# COMMAND ----------

df_predictions = lr_model.transform(scaled_df)
df_predictions.select("features", "label", "prediction").display()

# COMMAND ----------

scaled_df.printSchema()

# COMMAND ----------

trainingSummary = lr_model.summary

# Obtain the objective per iteration
objectiveHistory = trainingSummary.objectiveHistory
print("objectiveHistory:")
for objective in objectiveHistory:
    print(objective)

# for multiclass, we can inspect metrics on a per-label basis
print("False positive rate by label:")
for i, rate in enumerate(trainingSummary.falsePositiveRateByLabel):
    print("label %d: %s" % (i, rate))

print("True positive rate by label:")
for i, rate in enumerate(trainingSummary.truePositiveRateByLabel):
    print("label %d: %s" % (i, rate))

print("Precision by label:")
for i, prec in enumerate(trainingSummary.precisionByLabel):
    print("label %d: %s" % (i, prec))

print("Recall by label:")
for i, rec in enumerate(trainingSummary.recallByLabel):
    print("label %d: %s" % (i, rec))

print("F-measure by label:")
for i, f in enumerate(trainingSummary.fMeasureByLabel()):
    print("label %d: %s" % (i, f))

accuracy = trainingSummary.accuracy
falsePositiveRate = trainingSummary.weightedFalsePositiveRate
truePositiveRate = trainingSummary.weightedTruePositiveRate
fMeasure = trainingSummary.weightedFMeasure()
precision = trainingSummary.weightedPrecision
recall = trainingSummary.weightedRecall
print("Accuracy: %s\nFPR: %s\nTPR: %s\nF-measure: %s\nPrecision: %s\nRecall: %s"
      % (accuracy, falsePositiveRate, truePositiveRate, fMeasure, precision, recall))

# COMMAND ----------

class ModelTester:
    def __init__(self, df_scaled: DataFrame, df_unscaled: DataFrame,features_col="features", label_col="label", num_folds=3, seed=123):
        """
        Initialize ModelTester with scaled and unscaled DataFrames, label column, and cross-validation settings.
        """
        self.df_scaled = df_scaled
        self.df_unscaled = df_unscaled
        self.features_col = features_col
        self.label_col = label_col
        self.num_folds = num_folds
        self.seed = seed
        self.results = []

    def run_cv(self, model_class, param_dict, df):
        """
        Run cross-validation on the specified model with given hyperparameters.

        Parameters:
        - model_class: A Spark ML classifier (e.g., LogisticRegression)
        - param_dict: Dictionary of hyperparameters for the model
        - df: The DataFrame to train on (either scaled or unscaled)
        - features_col: Column name of features

        Returns:
        - Dictionary containing model, hyperparameters, and evaluation metrics
        """
        # Initialize model with feature and label columns
        model = model_class(featuresCol=self.features_col, labelCol=self.label_col)
        print(f"running eval on {model_class.__name__} with {param_dict}")
        # Create parameter grid
        param_grid = ParamGridBuilder()
        for param, values in param_dict.items():
            param_object = model.getParam(param)
            param_grid = param_grid.addGrid(param_object, values)
        param_grid = param_grid.build()
        
        # Define evaluator with F1 score
        evaluator = MulticlassClassificationEvaluator(labelCol=self.label_col, metricName="f1")
        cv = CrossValidator(estimator=model, estimatorParamMaps=param_grid, evaluator=evaluator,
                            numFolds=self.num_folds, seed=self.seed)

        # Fit the cross-validator
        cv_model = cv.fit(df)
        # Get the best model and make predictions
        best_model = cv_model.bestModel
        predictions = best_model.transform(df)
       # Evaluate metrics using the evaluator
        f1_score = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})
        accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
        weighted_precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
        weighted_recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})

        metrics = {
            "model": model_class.__name__,
            "hyperparameters": param_dict,
            "f1_score": f1_score,
            "accuracy": accuracy,
            "weighted_precision": weighted_precision,
            "weighted_recall": weighted_recall
        }
        
        return metrics


    def evaluate_models(self, model_params):
        """
        Run cross-validation for each model with configurations for both scaled and unscaled features.
        """
        # Test models with unscaled features
        for model_class, params in model_params["unscaled"]:
            result = self.run_cv(model_class, params, self.df_unscaled)
            self.results.append(result)
        
        # Test models with scaled features
        for model_class, params in model_params["scaled"]:
            result = self.run_cv(model_class, params, self.df_scaled)
            self.results.append(result)
        

    def get_results(self):
        """
        Returns all results as a list of dictionaries.
        """
        return self.results


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

model_tester = ModelTester(df_scaled=scaled_df, df_unscaled=unscaled_df)

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


