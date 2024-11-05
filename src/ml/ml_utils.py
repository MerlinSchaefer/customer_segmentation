import mlflow
import numpy as np

from abc import ABC, abstractmethod

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.sql import DataFrame
from pyspark.ml.classification import OneVsRest
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score



class ModelTesterBase(ABC):
    def __init__(self, label_col="label", num_folds=3, seed=13):
        self.label_col = label_col
        self.num_folds = num_folds
        self.seed = seed
        self.results = []

    @abstractmethod
    def run_cv(self, model_class, param_dict, df):
        """
        Run cross-validation on the specified model with given hyperparameters.
        Each implementation should handle cross-validation and evaluation.
        """
        pass

    def log_results(self, model_class_name, param_dict, metrics):
        """
        Log evaluation metrics to the results list.
        """
        # log metrics manually as autolog does not always capture everything
        #mlflow.log_metrics(metrics)
        self.results.append({
            "model": model_class_name,
            "hyperparameters": param_dict,
            **metrics
        })

    def get_results(self):
        return self.results




class SparkMLModelTester(ModelTesterBase):
    def __init__(self, df_scaled: DataFrame, df_unscaled: DataFrame,features_col="features", label_col="label", num_folds=3, seed=13):
        """
        Initialize ModelTester with scaled and unscaled DataFrames, label column, and cross-validation settings.
        """
        super().__init__(label_col, num_folds, seed)
        self.df_scaled = df_scaled
        self.df_unscaled = df_unscaled
        self.features_col = features_col

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
        print(f"running eval on {model_class.__name__} with {param_dict}")
        # Create parameter grid
        param_grid = ParamGridBuilder()
        for param, values in param_dict.items():
            param_object = model.getParam(param)
            param_grid = param_grid.addGrid(param_object, values)
        param_grid = param_grid.build()
        
        # Define evaluator and CrossValidator
        evaluator = MulticlassClassificationEvaluator(labelCol=self.label_col, metricName="f1")
        cv = CrossValidator(estimator=model, 
                            estimatorParamMaps=param_grid, 
                            evaluator=evaluator,
                            numFolds=self.num_folds, 
                            seed=self.seed)

        # Fit the cross-validator
        cv_model = cv.fit(df)
        # Extract metrics from cross-validation
        # cv_model.avgMetrics contains the average metric for each ParamMap in param_grid
        # We use the best model's metrics for consistency with scikit-learn's approach
        best_index = np.argmax(cv_model.avgMetrics)
        best_param_map = cv_model.getEstimatorParamMaps()[best_index]
        best_params = {param.name: value for param, value in best_param_map.items()}
        best_model = cv_model.bestModel

        # Transform the data with the best model to get predictions
        predictions = best_model.transform(df)

        # Calculate metrics using the evaluator and averaging over cross-validation folds
        f1_score =  evaluator.evaluate(predictions, {evaluator.metricName: "f1"})
        accuracy =  evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
        weighted_precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
        weighted_recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
        

        metrics = {
            "model": model_class.__name__,
            "hyperparameters": param_dict,
            "best_parameters": best_params,
            "f1_score": f1_score,
            "accuracy": accuracy,
            "weighted_precision": weighted_precision,
            "weighted_recall": weighted_recall
        }
        
        # Log the results
        self.log_results(model_class.__name__, best_params, metrics)
        
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
    

class SklearnModelTester(ModelTesterBase):
    def __init__(self, X, y, num_folds=3, seed=13):
        super().__init__(num_folds, seed)
        self.X = X
        self.y = y

    def run_cv(self, model_class, param_dict, X=None, y=None, cv_type="grid", n_iter=10):
        """
        Run cross-validation on the specified model with given hyperparameters.

        Args:
        - model_class: A scikit-learn classifier (e.g., LogisticRegression)
        - param_dict: Dictionary of hyperparameters for the model
        - X: The features to train on (optional)
        - y: The labels to train on (optional)
        - cv_type: Type of cross-validation to perform (optional, default="grid")
        """ 
        X = X if X is not None else self.X
        y = y if y is not None else self.y

        assert cv_type in ["grid", "random"], "Invalid cross-validation type"

        # Use StratifiedKFold to ensure balanced splits
        stratified_cv = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)

        # Create search object
        if cv_type == "grid":
            search = GridSearchCV(model_class(), param_dict, cv=stratified_cv, scoring='f1_weighted',
                                   n_jobs=-1, n_iter = n_iter)
        elif cv_type == "random":
            search = RandomizedSearchCV(model_class(), param_dict, cv=stratified_cv, scoring='f1_weighted', 
                                        n_jobs=-1, n_iter = n_iter)

        # Fit the search object
        search.fit(X, y)

        # Best model from cross-validation
        best_model = search.best_estimator_
        best_params = search.best_params_
        # Calculate metrics using cross_val_score
        metrics = {
            "model": model_class.__name__,
            "hyperparameters": param_dict,
            "best_parameters": best_params,
            "f1_score": np.mean(cross_val_score(best_model, X, y, cv=stratified_cv, scoring='f1_weighted')),
            "accuracy": np.mean(cross_val_score(best_model, X, y, cv=stratified_cv, scoring='accuracy')),
            "weighted_precision": np.mean(cross_val_score(best_model, X, y, cv=stratified_cv, scoring='precision_weighted')),
            "weighted_recall": np.mean(cross_val_score(best_model, X, y, cv=stratified_cv, scoring='recall_weighted')),
        }

        # Log the results
        self.log_results(model_class.__name__, best_params, metrics)
        return metrics

    def evaluate_models(self, model_params, cv_type="grid", n_iter=10):
        """
        Run cross-validation for each model with configurations.

        Args:
        - model_params: Dictionary containing model classes and parameter grids.
        - cv_type: Type of cross-validation to perform ("grid" or "random").
        """
        for model_class, params in model_params:
            self.run_cv(model_class, params, cv_type=cv_type, n_iter=n_iter)

class CatBoostModelTester(ModelTesterBase):
    def __init__(self, X, y, label_col="label", num_folds=3, seed=13):
        super().__init__(label_col, num_folds, seed)
        self.X = X
        self.y = y

    def run_cv(self,model_class, param_dict, X=None, y=None):
        """
        Run cross-validation on the CatBoostClassifier with given hyperparameters using CatBoost's built-in grid search.

        Args:
        - param_dict: Dictionary of hyperparameters for the model
        - X: The features to train on (optional)
        - y: The labels to train on (optional)
        """

        X = X if X is not None else self.X
        y = y if y is not None else self.y

        assert X is not None and y is not None, "Features (X) and labels (y) must not be None."
        mlflow.autolog()
        # Initialize CatBoostClassifier
        catboost_model = model_class(loss_function='MultiClass', random_seed=self.seed, verbose=False)

        # Perform grid search using CatBoost's built-in method
        grid_search_result = catboost_model.grid_search(param_dict, X=X, y=y, cv=self.num_folds, stratified=True, refit=True, verbose=False)

        # Best model and parameters
        best_params = grid_search_result['params']
        best_model = catboost_model

        # Make predictions with the best model
        y_pred = best_model.predict(X)

        # Calculate metrics
        metrics = {
            "model": "CatBoostClassifier",
            "hyperparameters": param_dict,
            "best_parameters": best_params,
            "f1_score": f1_score(y, y_pred, average='weighted'),
            "accuracy": accuracy_score(y, y_pred),
            "weighted_precision": precision_score(y, y_pred, average='weighted'),
            "weighted_recall": recall_score(y, y_pred, average='weighted'),
        }

        # Log the results
        self.log_results("CatBoostClassifier", best_params, metrics)
        return metrics

    def evaluate_models(self, model_params):
        """
        Run cross-validation for CatBoost with the given parameter grid.

        Args:
        - param_dict: Dictionary containing hyperparameters to search over.
        """
        for model_class, params in model_params:
            self.run_cv(model_class, params)

    def get_results(self):
        """
        Returns all results as a list of dictionaries.
        """
        return self.results
    



# helper function that we will use for getting latest version of a model
def get_latest_model_version(model_name):
    """Helper function to get latest model version"""
    client = mlflow.MlflowClient()
    model_version_infos = client.search_model_versions("name = '%s'" % model_name)
    return max([model_version_info.version for model_version_info in model_version_infos])