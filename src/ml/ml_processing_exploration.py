# Databricks notebook source
from pyspark.sql.types import IntegerType, BooleanType, StringType, DoubleType
from pyspark.sql.functions import col, count, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, VectorAssembler, RobustScaler, StringIndexer, OneHotEncoder
from pipeline_utils import DataFrameColumnManager, FeaturePipelineBuilder

# COMMAND ----------

# reading silver data as we may perform other imputation than the current dlt gold table
df = spark.sql("""SELECT 
                    gender,
                    ever_married,
                    age,
                    graduated,
                    profession,
                    work_experience,
                    spending_score,
                    family_size,
                    var_1
                    FROM customer_segmentation_dev.silver_training_customer_data""")

# COMMAND ----------

# custom preprocessing logic
df = df.fillna("Other", subset=["profession"]).fillna("No", subset=["ever_married"])

# COMMAND ----------

df.display()

# COMMAND ----------

train_df = df

# COMMAND ----------

# Initialize the DataFrameColumnManager class with the train and test DataFrames
df_manager = DataFrameColumnManager(train_df)

# Call the prepare_columns method to get the necessary information
column_info = df_manager.prepare_columns()
num_cols = column_info['num_cols']
string_cols = column_info['string_cols']
num_missing_cols = column_info['num_missing_cols']
string_missing_cols = column_info['string_missing_cols']

# COMMAND ----------

# MAGIC %md
# MAGIC # Imputation and Scaling

# COMMAND ----------

pipeline_builder = FeaturePipelineBuilder(num_cols, string_cols, num_missing_cols)
pipeline_scaling  = pipeline_builder.create_pipeline(scale_numerical=True)
pipeline_no_scaling = pipeline_builder.create_pipeline(scale_numerical=False)

# COMMAND ----------

pipeline_model_scaling = pipeline_scaling.fit(train_df)

# Transform both training_df  
train_transformed_scaled_df = pipeline_model_scaling.transform(train_df)



# COMMAND ----------

train_df.display()

# COMMAND ----------

pipeline_model_no_scaling = pipeline_no_scaling.fit(train_df)

train_transformed_no_scaled_df = pipeline_model_no_scaling.transform(train_df)


# COMMAND ----------

train_transformed_scaled_df.display()

# COMMAND ----------

train_transformed_no_scaled_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Save pipeline for use in DLT 

# COMMAND ----------

pipeline_model_no_scaling.write().overwrite().save("dbfs:/FileStore/customer_segmentation/pipelines/preprocessing/no_scaling")

# COMMAND ----------

pipeline_model_scaling.write().overwrite().save("dbfs:/FileStore/customer_segmentation/pipelines/preprocessing/scaling")
