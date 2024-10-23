from pyspark.sql.types import IntegerType, BooleanType, StringType, DoubleType
from pyspark.sql.functions import col, count, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, VectorAssembler, RobustScaler, StringIndexer, OneHotEncoder


class DataFrameColumnManager:
    """
    A class to manage and prepare columns in a DataFrame, including extracting and casting columns,
    and identifying missing values, adhering to the Single Responsibility Principle.
    """
    
    def __init__(self, df_train, df_test):
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
        for column in columns:
            self.df_train = self.df_train.withColumn(column, col(column).cast("double"))
            self.df_test = self.df_test.withColumn(column, col(column).cast("double"))

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