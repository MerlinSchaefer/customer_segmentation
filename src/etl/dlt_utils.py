import yaml
import pyspark.sql.types as T
from pyspark.sql import DataFrame
from pyspark.sql.functions import lit, monotonically_increasing_id
from databricks.sdk.runtime import *



# Function to get the data type of a column
def get_column_data_type(df, column):
    return [field.dataType for field in df.schema.fields if field.name == column][0]


def load_schema_from_yaml(yaml_path: str, schema_name: str) -> T.StructType:
    """
    Load a schema from a YAML file into Spark format.

    Args:
        yaml_path (str): Path to the YAML file.
        schema_name (str): Name of the schema to load. 
                            Must match the YAML.
    Returns:
        A Spark T.StructType object for the specified schema.
    """
    with open(yaml_path, 'r') as yaml_file:
        schemas = yaml.safe_load(yaml_file)
    
    # Extract the specified schema
    schema_fields = schemas[schema_name]
    
    # Convert the schema definition into a Spark T.StructType
    fields = [T.StructField(field['name'], getattr(T, field['type'])(), True) for field in schema_fields]
    
    return T.StructType(fields)


def load_yaml_config(config_path:str) -> dict:
    """
    Load a YAML configuration file into a dictionary

    Args:
        config_path (str): Path to the YAML file
    
    Returns:
        A dictionary of configuration values
    """
    with open(config_path, 'r') as config_file:
        return yaml.safe_load(config_file)

def create_lookup_df(mapping: dict, column_name: str) -> DataFrame:
    """
    Create a Spark DataFrame from a dictionary of values
    as a basis for a lookup table

    Args:
        mapping (dict): Dictionary of values to be used as lookup table
        column_name (str): Name of the column to be used as and identifier 
                            in the lookup table
    Returns:
        A Spark DataFrame with two columns: column and string_value
    """
    lookup_list = [(str(k), float(v), column_name) for k, v in mapping.items()]
        # Define an explicit schema
    schema = T.StructType([
        T.StructField("string_value", T.StringType(), True),
        T.StructField("numeric_value", T.DoubleType(), True),
        T.StructField("column", T.StringType(), True)
    ])
    return spark.createDataFrame(lookup_list, schema=schema).withColumn("column", lit(column_name))

def rowwise_join(df_1, df_2):
    """
    Joins two DataFrames rowwise, assuming the rows are aligned.
    Adds a row index to each DataFrame and joins on this index.
    
    Parameters:
    - df_1: DataFrame 1.
    - df_2: DataFrame 2.

    Returns:
    - A DataFrame with both 'features' and 'target' columns joined by index.
    """
    # Add a row index to both DataFrames
    df_1 = df_1.withColumn("row_index", monotonically_increasing_id())
    df_2 = df_2.withColumn("row_index", monotonically_increasing_id())

    # Perform the join on the row index
    df_combined = df_1.join(df_2, on="row_index").drop("row_index")
    
    return df_combined