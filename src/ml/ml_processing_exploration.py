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

# MAGIC %md
# MAGIC ## Train/Val* Split
# MAGIC
# MAGIC *there is a dedicated test set

# COMMAND ----------

train_df, test_df = df.randomSplit([0.8, 0.2], seed=13)

# COMMAND ----------



# # Get a list of integer & boolean columns
# integer_cols = [column.name for column in train_df.schema.fields if (column.dataType == IntegerType() or column.dataType == BooleanType())]

# # Loop through integer columns to cast each one to double
# for column in integer_cols:
#     train_df = train_df.withColumn(column, col(column).cast("double"))
#     test_df = test_df.withColumn(column, col(column).cast("double"))

# string_cols = [c.name for c in train_df.schema.fields if c.dataType == StringType()]
# num_cols = [c.name for c in train_df.schema.fields if c.dataType == DoubleType()]

# # Get a list of columns with missing values
# # Numerical
# num_missing_values_logic = [count(when(col(column).isNull(),column)).alias(column) for column in num_cols]
# row_dict_num = train_df.select(num_missing_values_logic).first().asDict()
# num_missing_cols = [column for column in row_dict_num if row_dict_num[column] > 0]

# # String
# string_missing_values_logic = [count(when(col(column).isNull(),column)).alias(column) for column in string_cols]
# row_dict_string = train_df.select(string_missing_values_logic).first().asDict()
# string_missing_cols = [column for column in row_dict_string if row_dict_string[column] > 0]

# print(f"Numeric columns with missing values: {num_missing_cols}")
# print(f"String columns with missing values: {string_missing_cols}")

# COMMAND ----------

# Initialize the DataFrameColumnManager class with the train and test DataFrames
df_manager = DataFrameColumnManager(train_df, test_df)

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

# # Imputer (median strategy for all double/numeric)
# to_impute = num_missing_cols
# imputer = Imputer(inputCols=to_impute, outputCols=to_impute, strategy='median')

# # Scale numerical
# numerical_assembler = VectorAssembler(inputCols=num_cols, outputCol="numerical_assembled")
# numerical_scaler = RobustScaler(inputCol="numerical_assembled", outputCol="numerical_scaled")

# # String/Cat Indexer (will encode missing/null as separate index)
# string_cols_indexed = [c + '_index' for c in string_cols]
# string_indexer = StringIndexer(inputCols=string_cols, outputCols=string_cols_indexed, handleInvalid="keep")

# # OHE categoricals
# ohe_cols = [column + '_ohe' for column in string_cols]
# one_hot_encoder = OneHotEncoder(inputCols=string_cols_indexed, outputCols=ohe_cols, handleInvalid="keep")

# # Assembler (All)
# feature_cols = ["numerical_scaled"] + ohe_cols
# vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# # Instantiate the pipeline in 2 variants
# # scaled and unscaled (for tree based models)
# pipeline_stages_scaling = [
#     imputer,
#     numerical_assembler,
#     numerical_scaler,
#     string_indexer,
#     one_hot_encoder,
#     vector_assembler
# ]


# pipeline_scaling = Pipeline(stages=pipeline_stages_scaling)


# COMMAND ----------

pipeline_model_scaling = pipeline_scaling.fit(train_df)

# Transform both training_df and test_df 
train_transformed_scaled_df = pipeline_model_scaling.transform(train_df)
test_transformed_scaled_df = pipeline_model_scaling.transform(test_df)


# COMMAND ----------


# # Version without scaling
# # Imputer (median strategy for all double/numeric)
# to_impute = num_missing_cols
# imputer = Imputer(inputCols=to_impute, outputCols=to_impute, strategy='median')

# # Assemble numerical features without scaling
# numerical_assembler = VectorAssembler(inputCols=num_cols, outputCol="numerical_assembled")

# # String/Cat Indexer (will encode missing/null as separate index)
# string_cols_indexed = [c + '_index' for c in string_cols]
# string_indexer = StringIndexer(inputCols=string_cols, outputCols=string_cols_indexed, handleInvalid="keep")

# # OneHotEncoder for categorical variables
# ohe_cols = [column + '_ohe' for column in string_cols]
# one_hot_encoder = OneHotEncoder(inputCols=string_cols_indexed, outputCols=ohe_cols, handleInvalid="keep")

# # Assembler for both numerical and categorical features
# feature_cols = ["numerical_assembled"] + ohe_cols
# vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# # Create the final pipeline
# pipeline = Pipeline(stages=[
#     imputer,
#     numerical_assembler,
#     string_indexer,
#     one_hot_encoder,
#     vector_assembler
# ])

# pipeline_no_scaling = Pipeline(stages=pipeline_stages_no_scaling)

# COMMAND ----------

pipeline_model_no_scaling = pipeline_no_scaling.fit(train_df)

train_transformed_no_scaled_df = pipeline_model_no_scaling.transform(train_df)
test_transformed_no_scaled_df = pipeline_model_no_scaling.transform(test_df)

# COMMAND ----------

train_transformed_scaled_df.display()

# COMMAND ----------

train_transformed_no_scaled_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Save pipeline for use in DLT 

# COMMAND ----------

pipeline_model_no_scaling.save("dbfs:/FileStore/customer_segmentation/pipelines/preprocessing/no_scaling")

# COMMAND ----------

pipeline_model_scaling.save("dbfs:/FileStore/customer_segmentation/pipelines/preprocessing/scaling")
