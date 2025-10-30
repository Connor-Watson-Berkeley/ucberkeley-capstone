# Databricks notebook source
# MAGIC %md
# MAGIC # ARIMA training
# MAGIC - This is an auto-generated notebook.
# MAGIC - To reproduce these results, attach this notebook to a cluster with runtime version **16.4.x-cpu-ml-scala2.12**, and rerun it.
# MAGIC - Compare trials in the [MLflow experiment](#mlflow/experiments/2255571809366092).
# MAGIC - Clone this notebook into your project folder by selecting **File > Clone** in the notebook toolbar.

# COMMAND ----------

import mlflow
import databricks.automl_runtime

target_col = "high"
time_col = "date"
unit = "day"


horizon = 10

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

import mlflow
import os
import uuid
import shutil
import pandas as pd
import pyspark.pandas as ps

# Create temp directory to download input data from MLflow
input_temp_dir = os.path.join(os.environ["SPARK_LOCAL_DIRS"], "tmp", str(uuid.uuid4())[:8])
os.makedirs(input_temp_dir)

# Download the artifact and read it into a pandas DataFrame
input_data_path = mlflow.artifacts.download_artifacts(run_id="3b626abacc5c4712bffdf63a64de53e7", artifact_path="data", dst_path=input_temp_dir)

input_file_path = os.path.join(input_data_path, "training_data")
input_file_path = "file://" + input_file_path
df_loaded = ps.from_pandas(pd.read_parquet(input_file_path))

# Preview data
display(df_loaded.head(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Aggregate data by `time_col`
# MAGIC Group the data by `time_col`, and take average if there are multiple `target_col` values in the same group.

# COMMAND ----------

group_cols = [time_col]
df_aggregated = df_loaded \
  .groupby(group_cols) \
  .agg(y=(target_col, "avg")) \
  .reset_index()
display(df_aggregated.head(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train ARIMA model
# MAGIC - Log relevant metrics to MLflow to track runs
# MAGIC - All the runs are logged under [this MLflow experiment](#mlflow/experiments/2255571809366092)
# MAGIC - Change the model parameters and re-run the training cell to log a different trial to the MLflow experiment

# COMMAND ----------

# Define the search space of seasonal period m
seasonal_periods = [1, 7]

# COMMAND ----------

result_columns = ["pickled_model", "mse", "rmse", "mae", "mape", "mdape", "smape", "coverage"]

def arima_training(history_pd):
  from databricks.automl_runtime.forecast.pmdarima.training import ArimaEstimator

  seasonal_period_list = seasonal_periods
 
  arima_estim = ArimaEstimator(horizon=horizon,
                               frequency_unit=unit,
                               metric="smape",
                               seasonal_periods=seasonal_period_list,
                               num_folds=20)

  results_pd = arima_estim.fit(history_pd)
 
  return results_pd[result_columns]

# COMMAND ----------

import mlflow
from databricks.automl_runtime.forecast.pmdarima.model import ArimaModel, mlflow_arima_log_model

with mlflow.start_run(experiment_id="2255571809366092", run_name="Arima") as mlflow_run:
  mlflow.set_tag("estimator_name", "ARIMA")

  df_aggregated = df_aggregated.rename(columns={time_col: "ds"})

  arima_results = arima_training(df_aggregated.to_pandas())
    
  # Log metrics to mlflow
  metric_name_map = {"mse": "mean_squared_error", "rmse": "root_mean_squared_error", "mae": "mean_absolute_error",
                     "mape": "mean_absolute_percentage_error", "mdape": "mdape", "smape": "smape", "coverage": "coverage"}
  
  avg_metrics = arima_results[metric_name_map.keys()].rename(columns=metric_name_map).mean().to_frame(name="mean_metrics").reset_index()
  avg_metrics["index"] = "val_" + avg_metrics["index"].astype(str)
  avg_metrics.set_index("index", inplace=True)
  mlflow.log_metrics(avg_metrics.to_dict()["mean_metrics"])
  
  # Save the model to mlflow
  pickled_model = arima_results["pickled_model"].to_list()[0]
  arima_model = ArimaModel(pickled_model, horizon, unit, df_aggregated["ds"].min(), df_aggregated["ds"].max(), time_col)

  # Generate sample input dataframe
  sample_input = df_loaded.tail(5).to_pandas()
  sample_input[time_col] = pd.to_datetime(sample_input[time_col])
  sample_input.drop(columns=[target_col], inplace=True)
  
  mlflow_arima_log_model(arima_model, sample_input=sample_input)

# COMMAND ----------

avg_metrics
# Uncomment to check the result details. By default, we only show the averaged metric becuase
# displaying the forecast result can take up a lot of storage for large datasets.
# arima_results.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analyze the predicted results

# COMMAND ----------

# Load the model
run_id = mlflow_run.info.run_id
loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

# COMMAND ----------

future_df = loaded_model._model_impl.python_model.make_future_dataframe()

# COMMAND ----------

# Predict future with the default horizon
forecast_pd = loaded_model._model_impl.python_model.predict_timeseries()

# COMMAND ----------

from databricks.automl_runtime.forecast.pmdarima.utils import plot

history_pd = df_aggregated.to_pandas()
# When visualizing, we ignore the first d (differencing order) points of the prediction results
# because it is impossible for ARIMA to predict the first d values
d = loaded_model._model_impl.python_model.model().order[1]
fig = plot(history_pd[d:], forecast_pd[d:])
fig

# COMMAND ----------

# MAGIC %md
# MAGIC ## Show the predicted results

# COMMAND ----------

predict_cols = ["ds", "yhat"]
forecast_pd = forecast_pd.reset_index()
display(forecast_pd[predict_cols].tail(horizon))