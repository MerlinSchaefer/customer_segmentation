# Databricks notebook source
import sys
sys.path.append("..")
import mlflow
import mlflow.pyfunc
import mlflow.data
import mlflow.sklearn
from mlflow.entities import ViewType

from pipeline_utils import load_and_prepare_ml_data, convert_to_sklearn_dataframe
from ml_utils import SklearnModelTester, SparkMLModelTester, CatBoostModelTester
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from hyperopt import hp
from pyspark.sql.functions import monotonically_increasing_id, col
from hyperopt import fmin, tpe, hp, Trials, SparkTrials, STATUS_OK
from hyperopt.pyll import scope
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
from pyspark.ml.classification import RandomForestClassifier
import numpy as np


# COMMAND ----------

# Create sparkml and sklearn dataframes
df_scaled, df_unscaled = load_and_prepare_ml_data()
X_scaled, y_scaled, X_unscaled, y_unscaled = convert_to_sklearn_dataframe(df_scaled, df_unscaled)

# COMMAND ----------

X_unscaled.display()

# COMMAND ----------

# Search space for CatBoostClassifier
catboost_space = {
    'iterations': scope.int(hp.quniform('iterations', 100, 500, 50)),
    'learning_rate': hp.uniform('learning_rate', 0.05, 0.25), 
    'depth': scope.int(hp.quniform('depth', 4, 12, 2)),
    'l2_leaf_reg': scope.int(hp.quniform('l2_leaf_reg', 1, 10, 1)),
}

# Search space for PySpark RandomForestClassifier
pyspark_rf_space = {
    'numTrees': scope.int(hp.quniform('num_trees', 10, 500, 25)),
    'maxDepth': scope.int(hp.quniform('max_depth', 3, 20, 1)),
    'maxBins': scope.int(hp.quniform('max_bins', 32, 128, 16)),
    "minInstancesPerNode": scope.int(hp.quniform("minInstancesPerNode",1, 20, 1))
}


# COMMAND ----------

def objective_catboost(params):


    # Start MLflow run
    with mlflow.start_run(nested=True) as mlflow_run:
        # Enable MLflow autologging
        mlflow.autolog(log_input_examples=True, silent=True)


        # Log parameters to MLflow
        mlflow.log_params(params)
        mlflow.log_dict(params, "params.json")
        # Stratified 3-Fold Cross-Validation
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=13)
        f1_scores = []


        for train_index, val_index in skf.split(X_unscaled, y_unscaled):
            X_train = X_unscaled.iloc[train_index]
            X_val = X_unscaled.iloc[val_index]
            y_train = y_unscaled.iloc[train_index]
            y_val = y_unscaled.iloc[val_index]


            # Train and validate using CatBoost
            model = CatBoostClassifier(**params, verbose=0, random_seed=13, early_stopping_rounds=30)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            # Calculate F1 score for the current fold and add it to the list
            f1 = f1_score(y_val, y_pred, average='weighted')
            f1_scores.append(f1)

        # Calculate the mean F1 score over all folds
        mean_f1_score = np.mean(f1_scores)
        # Log metrics to MLflow
        mlflow.log_metric("mean_f1_score", mean_f1_score)
        mlflow.catboost.log_model(model, "catboost_model")
        loss = -mean_f1_score  # We want to maximize F1, so minimize negative F1


    return {'loss': loss, 'status': STATUS_OK,"run": mlflow_run}


# COMMAND ----------

# Set up Trials for CatBoost
trials = Trials()
with mlflow.start_run(run_name="Catboost Model Tuning with Hyperopt") as parent_run:
    best_params_catboost = fmin(
        fn=objective_catboost,
        space=catboost_space,
        algo=tpe.suggest,
        max_evals=50,
        trials=trials,
        rstate=np.random.default_rng(13)
    )

# COMMAND ----------

best_result = trials.best_trial["result"]
best_run = best_result["run"]

# COMMAND ----------

display(best_run.__dict__)

# COMMAND ----------

display(best_result)

# COMMAND ----------

display(best_params_catboost)

# COMMAND ----------

# MAGIC %md
# MAGIC Start another round with params more centered around these.

# COMMAND ----------

catboost_space_fine_tune = {
    'iterations': scope.int(hp.quniform('iterations', 150, 300, 10)),  # Refine iterations range
    'learning_rate': hp.uniform('learning_rate', 0.02, 0.1),  # Narrow range around the best learning rate
    'depth': scope.int(hp.quniform('depth', 2, 6, 1)),  # Narrow range for depth
    'l2_leaf_reg': scope.int(hp.quniform('l2_leaf_reg', 8, 12, 1)),  # Narrow range for regularization
}

# COMMAND ----------

# Set up Trials for CatBoost
fine_tune_trials = Trials()
with mlflow.start_run(run_name="Catboost Model Tuning with Hyperopt") as parent_run:
    best_params_catboost = fmin(
        fn=objective_catboost,
        space=catboost_space_fine_tune,
        algo=tpe.suggest,
        max_evals=50,
        trials=fine_tune_trials,
        rstate=np.random.default_rng(13)
    )

# COMMAND ----------

# search over all runs
hpo_runs_catboost = mlflow.search_runs(
  experiment_ids=[parent_run.info.experiment_id],
  filter_string=f"tags.mlflow.parentRunId = '{parent_run.info.run_id}' AND attributes.status = 'FINISHED'",
  run_view_type=ViewType.ACTIVE_ONLY,
  order_by=["metrics.mean_f1_score DESC"]
)

display(hpo_runs_catboost)

# COMMAND ----------

hpo_runs_catboost

# COMMAND ----------

# Function for Spark models (e.g., PySpark RandomForestClassifier)
def objective_pyspark_rf(params):

    # Start MLflow run
    with mlflow.start_run(nested=True) as mlflow_run:
        mlflow.autolog(
        disable=False,
        log_input_examples=False,
        silent=True,
        exclusive=False)
        mlflow.log_params(params)
        mlflow.log_dict(params, "params.json")
        # Stratified 3-Fold Cross-Validation
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=13)
        f1_scores = []

        # Add a unique index to the Spark DataFrame
        df_unscaled_val = df_unscaled.withColumn("index", monotonically_increasing_id())

        # Convert the Spark DataFrame to Pandas to use StratifiedKFold (only indices are needed)
        df_pd = df_unscaled_val.select("index", "label").toPandas()
        for train_index, val_index in skf.split(df_pd['index'], df_pd['label']):
            # Get the indices for train and validation sets
            train_indices = df_pd['index'].iloc[train_index].tolist()
            val_indices = df_pd['index'].iloc[val_index].tolist()

            # Filter the Spark DataFrame based on train and validation indices
            train_df = df_unscaled_val.filter(col("index").isin(train_indices))
            val_df = df_unscaled_val.filter(col("index").isin(val_indices))

            # Train and validate using PySpark RandomForestClassifier
            model = RandomForestClassifier(featuresCol="features", labelCol="label",seed=13, **params)
            fitted_model = model.fit(train_df)
            predictions = fitted_model.transform(val_df)
            # Collect predictions and ground truth for F1 score calculation
            y_pred = [row.prediction for row in predictions.select("prediction").collect()]
            y_true = [row.label for row in val_df.select("label").collect()]


            # Calculate F1 score for the current fold and add it to the list
            f1 = f1_score(y_true, y_pred, average='weighted')
            f1_scores.append(f1)

        # Calculate the mean F1 score over all folds
        mean_f1_score = np.mean(f1_scores)
        loss = -mean_f1_score  # We want to maximize F1, so minimize negative F1

        # Log metrics to MLflow
        mlflow.log_metric("mean_f1_score", mean_f1_score)
        mlflow.spark.log_model(fitted_model, "spark_rf_model")

    return {'loss': loss, 'status': STATUS_OK, "run": mlflow_run}

# COMMAND ----------

# Set up SparkTrials for PySpark RandomForest
spark_trials = Trials()
with mlflow.start_run(run_name="PySpark RF Model Tuning with Hyperopt") as parent_run_spark:
    best_params_pyspark_rf = fmin(
        fn=objective_pyspark_rf,
        space=pyspark_rf_space,
        algo=tpe.suggest,
        max_evals=50,
        trials=spark_trials,
        rstate=np.random.default_rng(42)
    )

# COMMAND ----------

display(best_params_pyspark_rf)

# COMMAND ----------


hpo_runs_pd_spark = mlflow.search_runs(
  experiment_ids=[parent_run_spark.info.experiment_id],
  filter_string=f"tags.mlflow.parentRunId = '{parent_run.info.run_id}' AND attributes.status = 'FINISHED'",
  run_view_type=ViewType.ACTIVE_ONLY,
  order_by=["metrics.mean_f1_score DESC"]
)

display(hpo_runs_pd_spark)

