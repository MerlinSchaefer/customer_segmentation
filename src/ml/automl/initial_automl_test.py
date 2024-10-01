# Databricks notebook source
# MAGIC %md
# MAGIC ## Initial AutoML Test
# MAGIC
# MAGIC This simple auto ml classification without any previous manual feature preparation or engineering can serve as a baseline for future modelling.

# COMMAND ----------


from databricks import automl

# column 'Segmentation' as the label
# load the data from table 
df = spark.table("customer_segmentation.train_raw")



summary = automl.classify(
    dataset=df,
    target_col="Segmentation",
    timeout_minutes=60,  # Optional: Set time limit for AutoML run, adjust as necessary
    primary_metric="f1",  
)

# The 'summary' object contains information about the best model and performance metrics
# You can explore it further to retrieve the best model or view details


# COMMAND ----------

print(summary.best_trial)

# COMMAND ----------


