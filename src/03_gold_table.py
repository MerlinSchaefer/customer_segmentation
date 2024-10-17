# Databricks notebook source
import dlt
from pyspark.sql.functions import when, col, lit, create_map, desc
from databricks.sdk.runtime import *
from dlt_utils import create_lookup_df, load_yaml_config

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

# Gold Table: Business-ready or aggregated data
@dlt.table
def gold_customer_features():
    silver_df = dlt.read("silver_training_customer_data")
    gold_df = silver_df.drop("id", "segmentation","inserted_at")
    # custom imputation for missing values of profession and ever_married
    gold_df = gold_df.fillna("Other", subset=["profession"]).fillna("No", subset=["ever_married"])

    for mapping_entry in custom_mappings:
        mapping = mapping_entry['mapping']
        columns = mapping_entry['columns']

        # Apply mapping for each column in the list
        for column in columns:
            # Apply the mapping using when() and lit(), chaining the conditions for each mapping key
            col_expr = None
            for key, value in mapping.items():
                # Add a new when() clause for each mapping
                if col_expr is None:
                    col_expr = when(col(column) == key, lit(value))
                else:
                    col_expr = col_expr.when(col(column) == key, lit(value))

            # Final fallback if no mapping is found (set to None or leave the column as is)
            col_expr = col_expr.otherwise(lit(None))

            # Update the DataFrame with the transformed column
            gold_df = gold_df.withColumn(column, col_expr)

    # Mode Imputation for all categorical columns
    for column in categorical_columns_custom:
        mode_value_row = gold_df.groupBy(column).count().orderBy(desc("count")).first()
        mode_value = mode_value_row[0]
        mode_value = str(mode_value) if isinstance(mode_value, bool) else mode_value
        gold_df = gold_df.withColumn(column, when(col(column).isNull(), lit(mode_value)).otherwise(col(column)))



    # Median Imputation for all numerical columns
    for column in numerical_columns:
        median_value = gold_df.approxQuantile(column, [0.5], 0.01)[0]
        median_value = float(median_value)
        gold_df = gold_df.withColumn(column, when(col(column).isNull(), lit(median_value)).otherwise(col(column)))


    return gold_df

# COMMAND ----------

# ##DEBUG
# config_path = "../resources/gold_table_config.yml"
# gold_table_config  = load_yaml_config(config_path)
# # Define categorical columns for mode imputation
# # Extracting the column lists
# categorical_columns_custom = gold_table_config['categorical_columns']['custom']
# categorical_columns_onehot = gold_table_config['categorical_columns']['onehot']
# categorical_columns_ordinal = gold_table_config['categorical_columns']['ordinal']

# # Extracting numerical columns
# numerical_columns = gold_table_config['numerical_columns']

# # Extracting custom mappings
# custom_mappings = gold_table_config['custom_mappings']
# custom_mapping_target = gold_table_config['custom_mapping_target']


# bronze_df = spark.read.table("customer_segmentation.train_raw")
# # Standardize column names (lowercase, underscores instead of spaces)
# df = bronze_df.toDF(*[col.lower().replace(' ', '_') for col in bronze_df.columns])

# # Convert Age and Family Size to integers, Work Experience to float
# silver_df = (df.withColumn("age", col("age").cast("int"))
#         .withColumn("family_size", col("family_size").cast("int"))
#         .withColumn("work_experience", col("work_experience").cast("float"))
#         )

# gold_df = silver_df.drop("id", "segmentation","inserted_at")
# # custom imputation for missing values of profession and ever_married
# gold_df = gold_df.fillna("Other", subset=["profession"]).fillna("No", subset=["ever_married"])





# display(gold_df)


# COMMAND ----------

custom_mappings

# COMMAND ----------

# from pyspark.sql.functions import col, lit, create_map, when

# # Assuming custom_mappings and gold_df are defined
#     # Loop through each mapping and apply to respective columns

# # Loop through each mapping and apply to respective columns
# for mapping_entry in custom_mappings:
#     mapping = mapping_entry['mapping']
#     columns = mapping_entry['columns']

#     # Apply mapping for each column in the list
#     for column in columns:
#         # Apply the mapping using when() and lit(), chaining the conditions for each mapping key
#         col_expr = None
#         for key, value in mapping.items():
#             # Add a new when() clause for each mapping
#             if col_expr is None:
#                 col_expr = when(col(column) == key, lit(value))
#             else:
#                 col_expr = col_expr.when(col(column) == key, lit(value))

#         # Final fallback if no mapping is found (set to None or leave the column as is)
#         col_expr = col_expr.otherwise(lit(None))

#         # Update the DataFrame with the transformed column
#         gold_df = gold_df.withColumn(column, col_expr)

# COMMAND ----------

# # Mode Imputation for all categorical columns
# for column in categorical_columns_custom:
#     mode_value_row = gold_df.groupBy(column).count().orderBy(desc("count")).first()
#     mode_value = mode_value_row[0]
#     mode_value = str(mode_value) if isinstance(mode_value, bool) else mode_value
#     gold_df = gold_df.withColumn(column, when(col(column).isNull(), lit(mode_value)).otherwise(col(column)))



# # Median Imputation for all numerical columns
# for column in numerical_columns:
#     median_value = gold_df.approxQuantile(column, [0.5], 0.01)[0]
#     median_value = float(median_value)
#     gold_df = gold_df.withColumn(column, when(col(column).isNull(), lit(median_value)).otherwise(col(column)))

