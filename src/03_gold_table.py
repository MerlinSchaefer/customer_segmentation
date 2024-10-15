# Databricks notebook source
import dlt
from pyspark.sql.functions import when, col, lit, create_map, desc
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler

from dlt_utils import create_lookup_df, load_yaml_config

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
    for mapping, columns in custom_mapping:
        for column in columns:
            if lookup_df is None:
                lookup_df = create_lookup_df(mapping, column)
            else:
                lookup_df = lookup_df.union(create_lookup_df(mapping, column))
    
    return lookup_df

# COMMAND ----------

# Gold Table: Business-ready or aggregated data
@dlt.table
def gold_customer_features():
    silver_df = dlt.read("silver_training_customer_data")
    gold_df = silver_df.drop("id", "segmentation","inserted_at")
    # custom imputation for missing values of profession and ever_married
    gold_df.fillna("Other", subset=["profession"]).fillna("No", subset=["ever_married"])
    
    # encoding of categorical variables
    # Custom encoding for Ever_Married, Graduated, Gender, and Spending_Score
    for mapping_entry in custom_mapping:
            mapping = mapping_entry['mapping']
            columns = mapping_entry['columns']
            for column in columns:
                gold_df = gold_df.withColumn(column, 
                    when(col(column).isin(mapping.keys()), 
                        create_map([lit(k), lit(v) for k, v in mapping.items()])[col(column)]
                    ).otherwise(lit(None)))  # Handle invalid values by setting to None

        # Mode Imputation for all categorical columns
        for column in categorical_columns_custom:
            mode_value = gold_df.groupBy(column).count().orderBy(desc("count")).first()[0]  # Calculate mode
            gold_df = gold_df.withColumn(column, when(col(column).isNull(), lit(mode_value)).otherwise(col(column)))

        # Median Imputation for all numerical columns
        for column in numerical_columns:
            median_value = gold_df.approxQuantile(column, [0.5], 0.01)[0]  # Calculate median
            gold_df = gold_df.withColumn(column, when(col(column).isNull(), lit(median_value)).otherwise(col(column)))

        return gold_df

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
