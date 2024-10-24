# Databricks notebook source
from pyspark.ml.classification import (LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier, MultilayerPerceptronClassifier, LinearSVC)
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline

# COMMAND ----------

scaled_features = spark.read.table("customer_segmentation_dev.gold_scaled_features")
unscaled_features = spark.read.table("customer_segmentation_dev.gold_no_scaled_features")
target = spark.read.table("customer_segmentation_dev.gold_customer_ml_target")

# COMMAND ----------

scaled_features.display()

# COMMAND ----------

target.display()

# COMMAND ----------

scaled_df = scaled_features.join(target, "id").withColumnRenamed("segmentation", "label").withColumn("label", F.col("label").cast("double"))
unscaled_df = unscaled_features.join(target, "id").withColumnRenamed("segmentation", "label").withColumn("label", F.col("label").cast("double"))

# COMMAND ----------

scaled_df.display()

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
