import yaml
from pyspark.sql import types as T
from pyspark.sql import DataFrame

def load_schema_from_yaml(yaml_path: str, schema_name: str) -> T.StructType:
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
    lookup_list = [(k, v) for k, v in mapping.items()]
    return spark.createDataFrame(lookup_list, schema=["string_value", "numeric_value"]).withColumn("column", F.lit(column_name))