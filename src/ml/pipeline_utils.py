import pandas as pd
from pyspark.sql.types import IntegerType, BooleanType, StringType, DoubleType
from pyspark.sql.functions import col, count, when, udf
from pyspark.sql import DataFrame
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, VectorAssembler, RobustScaler, StringIndexer, OneHotEncoder
from pyspark.ml.linalg import SparseVector, DenseVector
from pyspark.ml.linalg import VectorUDT
from databricks.sdk.runtime import spark

class DataFrameColumnManager:
    """
    A class to manage and prepare columns in a DataFrame, including extracting and casting columns,
    and identifying missing values, adhering to the Single Responsibility Principle.
    """
    
    def __init__(self, df_train, df_test = None):
        """
        Initialize the class with training and testing DataFrames.
        
        Parameters:
        - df_train: The training DataFrame.
        - df_test: The testing DataFrame.
        """
        self.df_train = df_train
        self.df_test = df_test
    
    def get_integer_and_boolean_columns(self):
        """
        Extract integer and boolean columns from the training DataFrame.
        
        Returns:
        - List of integer and boolean columns.
        """
        return [column.name for column in self.df_train.schema.fields if isinstance(column.dataType, (IntegerType, BooleanType))]

    def cast_columns_to_double(self, columns):
        """
        Cast the specified columns in both training and testing DataFrames to double type.
        
        Parameters:
        - columns: List of columns to cast.
        
        Updates:
        - Modifies both self.df_train and self.df_test by casting the specified columns to double.
        """
        if self.df_test:
            for column in columns:
                self.df_train = self.df_train.withColumn(column, col(column).cast("double"))
                self.df_test = self.df_test.withColumn(column, col(column).cast("double"))
        else:
            for column in columns:
                self.df_train = self.df_train.withColumn(column, col(column).cast("double"))


    def get_string_columns(self):
        """
        Extract string columns from the training DataFrame.
        
        Returns:
        - List of string columns.
        """
        return [column.name for column in self.df_train.schema.fields if isinstance(column.dataType, StringType)]

    def get_double_columns(self):
        """
        Extract double columns from the training DataFrame.
        
        Returns:
        - List of double columns.
        """
        return [column.name for column in self.df_train.schema.fields if isinstance(column.dataType, DoubleType)]

    def get_columns_with_missing_values(self, columns):
        """
        Identify columns with missing values from a given list of columns.
        
        Parameters:
        - columns: List of columns to check for missing values.
        
        Returns:
        - List of columns that have missing values.
        """
        missing_values_logic = [count(when(col(column).isNull(), column)).alias(column) for column in columns]
        row_dict = self.df_train.select(missing_values_logic).first().asDict()
        return [column for column in row_dict if row_dict[column] > 0]

    def prepare_columns(self):
        """
        High-level function to prepare and cast necessary columns, and identify columns with missing values.
        
        Returns:
        - A dictionary with the following keys:
          - 'num_cols': List of numerical columns.
          - 'string_cols': List of string columns.
          - 'num_missing_cols': List of numerical columns with missing values.
          - 'string_missing_cols': List of string columns with missing values.
        """
        # Get integer and boolean columns
        integer_cols = self.get_integer_and_boolean_columns()
        
        # Cast integer and boolean columns to double in both train and test DataFrames
        self.cast_columns_to_double(integer_cols)
        
        # Get string and numerical columns
        string_cols = self.get_string_columns()
        num_cols = self.get_double_columns()
        
        # Get columns with missing values
        num_missing_cols = self.get_columns_with_missing_values(num_cols)
        string_missing_cols = self.get_columns_with_missing_values(string_cols)
        
        return {
            'num_cols': num_cols,
            'string_cols': string_cols,
            'num_missing_cols': num_missing_cols,
            'string_missing_cols': string_missing_cols
        }


class FeaturePipelineBuilder:
    def __init__(self, num_cols, string_cols, num_missing_cols):
        self.num_cols = num_cols
        self.string_cols = string_cols
        self.num_missing_cols = num_missing_cols

    def create_pipeline(self, scale_numerical=False):
        # Step 1: Imputer (median strategy for all double/numeric columns)
        imputer = Imputer(inputCols=self.num_missing_cols, outputCols=self.num_missing_cols, strategy='median')

        # Step 2: Numerical features (with or without scaling)
        numerical_assembler = VectorAssembler(inputCols=self.num_cols, outputCol="numerical_assembled")
        
        # Conditionally add the scaling step
        if scale_numerical:
            numerical_output_col = "numerical_scaled"
            numerical_scaler = RobustScaler(inputCol="numerical_assembled", outputCol=numerical_output_col)
        else:
            numerical_output_col = "numerical_assembled"

        # Step 3: String/Cat Indexer (encode missing/null as separate index)
        string_cols_indexed = [c + '_index' for c in self.string_cols]
        string_indexer = StringIndexer(inputCols=self.string_cols, outputCols=string_cols_indexed, handleInvalid="keep")

        # Step 4: OneHotEncoder for categorical variables
        ohe_cols = [col + '_ohe' for col in self.string_cols]
        one_hot_encoder = OneHotEncoder(inputCols=string_cols_indexed, outputCols=ohe_cols, handleInvalid="keep")

        # Step 5: Assemble all features (numerical + categorical)
        feature_cols = [numerical_output_col] + ohe_cols
        vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

        # Build the pipeline with the necessary steps
        stages = [imputer, numerical_assembler, string_indexer, one_hot_encoder, vector_assembler]
        
        # Add the scaling step if needed
        if scale_numerical:
            stages.insert(2, numerical_scaler)

        # Return the final pipeline
        return Pipeline(stages=stages)
    
def adjust_labels(df: DataFrame, label_col: str="label") -> DataFrame:
    """
    Adjusts labels from 1-4 to 0-3 for compatibility with Spark ML models.
    
    Parameters:
    - df: DataFrame containing the label column.
    - label_col: Name of the label column to be adjusted.

    Returns:
    - DataFrame with labels adjusted to be in the range [0, 3].
    """
    adjusted_df = df.withColumn(label_col, when(col(label_col) == 1, 0)
                                           .when(col(label_col) == 2, 1)
                                           .when(col(label_col) == 3, 2)
                                           .when(col(label_col) == 4, 3))
    return adjusted_df



# Apply the UDF to the features column to convert to dense vectors
def convert_sparse_to_dense(df:DataFrame, features_col="features") -> DataFrame:
    """
    Convert the features column from sparse vectors to dense vectors.
    """
    # UDF to convert sparse vectors to dense vectors
    def sparse_to_dense(vector):
        if isinstance(vector, SparseVector):
            return DenseVector(vector.toArray())
        else:
            return vector

    # Registering the UDF to convert the vector
    sparse_to_dense_udf = udf(sparse_to_dense, VectorUDT())
    return df.withColumn(features_col, sparse_to_dense_udf(df[features_col]))

def convert_to_sklearn_dataframe(features_scaled:DataFrame,features_unscaled:DataFrame) -> (pd.DataFrame,pd.Series,pd.DataFrame, pd.Series):
    """
    Convert the features and labels to a Pandas DataFrame for compatibility with sklearn models.
    """
    # TODO: refactor to generic conversion not scaled and unscaled

    # Convert the label and features to Pandas DataFrame
    # Extract labels as a list
    labels_scaled = features_scaled.select("label").rdd.flatMap(lambda x: x).collect()
    labes_unscaled = features_unscaled.select("label").rdd.flatMap(lambda x: x).collect()
    # Extract features as a list of lists (where each row is a feature vector)
    features_scaled = features_scaled.select("features").rdd \
        .map(lambda row: row.features.toArray()) \
        .collect()
    features_unscaled = features_unscaled.select("features").rdd \
        .map(lambda row: row.features.toArray()) \
            .collect()
    # Create a Pandas DataFrame from the features and labels
    pandas_df_scaled = pd.DataFrame(features_scaled)
    pandas_df_scaled['label'] = labels_scaled
    pandas_df_unscaled = pd.DataFrame(features_unscaled)
    pandas_df_unscaled['label'] = labes_unscaled


    y_scaled = pandas_df_scaled['label']
    X_scaled = pandas_df_scaled.drop('label', axis=1)
    y_unscaled = pandas_df_unscaled['label']
    X_unscaled = pandas_df_unscaled.drop('label', axis=1)
    return X_scaled, y_scaled, X_unscaled, y_unscaled


def load_and_prepare_ml_data(schema_name = "customer_segmentation_dev",
                             scaled_table_name = "gold_scaled_features", 
                             unscaled_table_name = "gold_no_scaled_features", 
                             target_table_name = "gold_customer_ml_target") -> (DataFrame,DataFrame):
    """
    Convenience function to load and prepare the ML data for training.
    Bundles the above functions into a single call.
    """
    scaled_features = spark.read.table(f"{schema_name}.{scaled_table_name}")
    unscaled_features = spark.read.table(f"{schema_name}.{unscaled_table_name}")
    target = spark.read.table(f"{schema_name}.{target_table_name}")

    target = adjust_labels(target, label_col="segmentation")
    scaled_df = scaled_features.join(target, "id").withColumnRenamed("segmentation", "label").withColumn("label", col("label").cast("double"))
    unscaled_df = unscaled_features.join(target, "id").withColumnRenamed("segmentation", "label").withColumn("label", col("label").cast("double"))
    df_scaled = convert_sparse_to_dense(scaled_df)
    df_unscaled = convert_sparse_to_dense(unscaled_df)
    return df_scaled, df_unscaled