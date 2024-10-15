import yaml
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.sql import DataFrame
from pyspark.sql.functions import lit
from databricks.sdk.runtime import *

def load_schema_from_yaml(yaml_path: str, schema_name: str) -> StructType:
    """
    Load a schema from a YAML file into Spark format.

    Args:
        yaml_path (str): Path to the YAML file.
        schema_name (str): Name of the schema to load. 
                            Must match the YAML.
    Returns:
        A Spark StructType object for the specified schema.
    """
    with open(yaml_path, 'r') as yaml_file:
        schemas = yaml.safe_load(yaml_file)
    
    # Extract the specified schema
    schema_fields = schemas[schema_name]
    
    # Convert the schema definition into a Spark StructType
    fields = [StructField(field['name'], getattr(T, field['type'])(), True) for field in schema_fields]
    
    return StructType(fields)


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
    lookup_list = [(k, v) for k, v in mapping.items()]
        # Define an explicit schema
    schema = StructType([
        StructField("string_value", StringType(), True),
        StructField("numeric_value", DoubleType(), True),
        StructField("column", StringType(), True)
    ])
    return spark.createDataFrame(lookup_list, schema=schema).withColumn("column", lit(column_name))