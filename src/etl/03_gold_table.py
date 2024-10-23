# Databricks notebook source
import dlt
from pyspark.sql.functions import when, col, lit, desc
from pyspark.sql.types import IntegerType, FloatType
from pyspark.ml import PipelineModel
from databricks.sdk.runtime import *
from dlt_utils import create_lookup_df, load_yaml_config, get_column_data_type

from databricks.feature_store import FeatureStoreClient


# COMMAND ----------

fs = FeatureStoreClient()

# COMMAND ----------

spark.read.format("csv").option("header", "true").load("/FileStore/customer_segmentation/train/").show()


# COMMAND ----------

config_path = spark.conf.get("gold_table_config_path")
gold_table_config  = load_yaml_config(config_path)

# COMMAND ----------

# Extracting the column lists
categorical_columns_custom = gold_table_config['categorical_columns']['custom']
categorical_columns_onehot = gold_table_config['categorical_columns']['onehot']
categorical_columns_ordinal = gold_table_config['categorical_columns']['ordinal']

# Extracting numerical columns
numerical_columns = gold_table_config['numerical_columns']

# Extracting custom mappings
custom_mappings = gold_table_config['custom_mappings']
custom_mapping_target = gold_table_config['custom_mapping_target']

# COMMAND ----------

@dlt.table
def gold_mapping_lookup():
    # Create an empty DataFrame to store all lookup mappings
    lookup_df = None
    
    # Iterate through custom mappings and create a DataFrame for each, then union them
    for mapping_entry in custom_mappings:
        mapping = mapping_entry['mapping']
        columns = mapping_entry['columns']
        for column in columns:
            if lookup_df is None:
                lookup_df = create_lookup_df(mapping, column)
            else:
                lookup_df = lookup_df.union(create_lookup_df(mapping, column))
    
    return lookup_df

# COMMAND ----------


@dlt.table
def gold_customer_ml_target():
    gold_df_target = dlt.read("silver_training_customer_data").select("segmentation")

    # Custom encoding for segmentation (target column)
    gold_df_target = gold_df_target.withColumn("segmentation", 
                       when(col("segmentation") == "A", custom_mapping_target["A"])
                        .when(col("segmentation") == "B", custom_mapping_target["B"])
                        .when(col("segmentation") == "C", custom_mapping_target["C"])
                        .otherwise(custom_mapping_target["D"])
    )
    
    return gold_df_target

# @dlt.table
# def debug_gold_customer_features():
#     silver_df = dlt.read("silver_training_customer_data")
#     gold_df = silver_df.drop("id", "segmentation", "inserted_at")
#     gold_df = gold_df.fillna("Other", subset=["profession"]).fillna("No", subset=["ever_married"])
    
#     # Debugging by outputting the intermediate gold_df
#     return gold_df


@dlt.table
def gold_customer_complete():
    silver_df = dlt.read("silver_training_customer_data")
    gold_df = silver_df.drop("id", "segmentation","inserted_at")
    # custom imputation for missing values of profession and ever_married
    gold_df = gold_df.fillna("Other", subset=["profession"]).fillna("No", subset=["ever_married"])

    for mapping_entry in custom_mappings:
        mapping = mapping_entry['mapping']
        columns = mapping_entry['columns']

        # Apply mapping for each column in the list
        for column in columns:
            col_expr = None
            for key, value in mapping.items():
                # Add a new when() clause for each mapping
                if col_expr is None:
                    col_expr = when(col(column) == key, lit(value))
                else:
                    col_expr = col_expr.when(col(column) == key, lit(value))

            # Final fallback if no mapping is found (set to None or leave the column as is)
            col_expr = col_expr.otherwise(lit(None))
            gold_df = gold_df.withColumn(column, col_expr)


    # Mode Imputation for all categorical columns
    for column in categorical_columns_custom:
        # Calculate mode safely by checking if there are valid results
        mode_value_row = gold_df.groupBy(column).count().orderBy(desc("count")).first()
        
        if mode_value_row is not None and mode_value_row[0] is not None:
            # Get the mode value
            mode_value = mode_value_row[0]
            # Convert booleans to string, if needed
            mode_value = str(mode_value) if isinstance(mode_value, bool) else mode_value
            # Apply mode imputation
            gold_df = gold_df.withColumn(column, when(col(column).isNull(), lit(mode_value)).otherwise(col(column)))
        else:
            print(f"No mode value found for column {column}. Skipping imputation.")

    # Median Imputation for all numerical columns
    for column in numerical_columns:
        # Get the original data type of the column
        original_type = get_column_data_type(gold_df, column)

        # Calculate median safely
        median_values = gold_df.approxQuantile(column, [0.5], 0.01)
        
        if median_values and len(median_values) > 0:  # Ensure the list is not empty
            # Convert to float and apply median imputation
            median_value = float(median_values[0])
            gold_df = gold_df.withColumn(column, when(col(column).isNull(), lit(median_value)).otherwise(col(column)))
            
            # After imputation, cast the column back to its original data type
            if isinstance(original_type, IntegerType):
                gold_df = gold_df.withColumn(column, col(column).cast("int"))
            elif isinstance(original_type, FloatType):
                gold_df = gold_df.withColumn(column, col(column).cast("float"))
        else:
            print(f"No median value found for column {column}. Skipping imputation.")



    return gold_df

# COMMAND ----------

# TODO: MOVE PATH TO CONFIG
spark.conf.get("pipeline_path")

# COMMAND ----------

@dlt.table
def gold_scaled_features():
    silver_df = dlt.read("silver_training_customer_data")
    gold_df = silver_df.drop("id", "segmentation","inserted_at")
    # custom imputation for missing values of profession and ever_married
    gold_df = gold_df.fillna("Other", subset=["profession"]).fillna("No", subset=["ever_married"])

    # Load the pipeline
  
    pipeline_model_scaling = PipelineModel.load(f"{pipeline_path}/scaling")

    gold_df_transformed = pipeline_model_scaling.transform(gold_df)

    return gold_df_transformed.select("features")

@dlt.table
def gold_no_scaled_features():
    silver_df = dlt.read("silver_training_customer_data")
    gold_df = silver_df.drop("id", "segmentation","inserted_at")
    # custom imputation for missing values of profession and ever_married
    gold_df = gold_df.fillna("Other", subset=["profession"]).fillna("No", subset=["ever_married"])

    # Load the pipeline
  
    pipeline_model_no_scaling = PipelineModel.load(f"{pipeline_path}/no_scaling")

    gold_df_transformed = pipeline_model_no_scaling.transform(gold_df)
        
    return gold_df_transformed.select("features")

