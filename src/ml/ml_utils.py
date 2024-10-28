from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.sql import DataFrame
from pyspark.ml.classification import OneVsRest

class ModelTester:
    def __init__(self, df_scaled: DataFrame, df_unscaled: DataFrame,features_col="features", label_col="label", num_folds=3, seed=123):
        """
        Initialize ModelTester with scaled and unscaled DataFrames, label column, and cross-validation settings.
        """
        self.df_scaled = df_scaled
        self.df_unscaled = df_unscaled
        self.label_col = label_col
        self.num_folds = num_folds
        self.seed = seed
        self.results = []

    def run_cv(self, model_class, param_dict, df):
        """
        Run cross-validation on the specified model with given hyperparameters.

        Parameters:
        - model_class: A Spark ML classifier (e.g., LogisticRegression)
        - param_dict: Dictionary of hyperparameters for the model
        - df: The DataFrame to train on (either scaled or unscaled)
        - features_col: Column name of features

        Returns:
        - Dictionary containing model, hyperparameters, and evaluation metrics
        """
        # Initialize model with feature and label columns
        model = model_class(featuresCol=self.features_col, labelCol=self.label_col)

        # Create parameter grid
        param_grid = ParamGridBuilder()
        for param, values in param_dict.items():
            param_grid = param_grid.addGrid(getattr(model, param), values)
        param_grid = param_grid.build()
        
        # Define evaluator and CrossValidator
        evaluator = MulticlassClassificationEvaluator(labelCol=self.label_col, metricName="accuracy")
        cv = CrossValidator(estimator=model, estimatorParamMaps=param_grid, evaluator=evaluator,
                            numFolds=self.num_folds, seed=self.seed)

        # Fit the cross-validator
        cv_model = cv.fit(df)

        # Extract model summary and evaluation metrics
        summary = cv_model.bestModel.summary
        metrics = {
            "model": model_class.__name__,
            "hyperparameters": param_dict,
            "accuracy": summary.accuracy,
            "weighted_precision": summary.weightedPrecision,
            "weighted_recall": summary.weightedRecall,
            "f_measure": summary.weightedFMeasure(),
            "false_positive_rate": summary.weightedFalsePositiveRate,
            "true_positive_rate": summary.weightedTruePositiveRate,
            "objective_history": summary.objectiveHistory,
            "precision_by_label": summary.precisionByLabel,
            "recall_by_label": summary.recallByLabel,
            "f_measure_by_label": summary.fMeasureByLabel()
        }
        
        # Add the metrics to results
        self.results.append(metrics)
        
        return metrics

    def evaluate_models(self, model_params):
        """
        Run cross-validation for each model with configurations for both scaled and unscaled features.
        """
        # Test models with scaled features
        for model_class, params in model_params["scaled"]:
            self.run_cv(model_class, params, self.df_scaled)
        
        # Test models with unscaled features
        for model_class, params in model_params["unscaled"]:
            self.run_cv(model_class, params, self.df_unscaled)
        
    def get_results(self):
        """
        Returns all results as a list of dictionaries.
        """
        return self.results
