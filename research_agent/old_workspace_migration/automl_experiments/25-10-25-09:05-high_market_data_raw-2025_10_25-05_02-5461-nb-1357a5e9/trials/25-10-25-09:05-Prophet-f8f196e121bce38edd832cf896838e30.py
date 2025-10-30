# Databricks notebook source
# MAGIC %md
# MAGIC # Prophet training
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
# MAGIC ## Train Prophet model
# MAGIC - Log relevant metrics to MLflow to track runs
# MAGIC - All the runs are logged under [this MLflow experiment](#mlflow/experiments/2255571809366092)
# MAGIC - Change the model parameters and re-run the training cell to log a different trial to the MLflow experiment

# COMMAND ----------

import logging

# disable informational messages from prophet
logging.getLogger("py4j").setLevel(logging.WARNING)

# COMMAND ----------

result_columns = ["model_json", "mse", "rmse", "mae", "mape", "mdape", "smape", "coverage"]

def prophet_training(history_pd):
  from hyperopt import hp
  from databricks.automl_runtime.forecast.prophet.forecast import ProphetHyperoptEstimator

  seasonality_mode = ["additive", "multiplicative"]
  search_space =  {
    "changepoint_prior_scale": hp.loguniform("changepoint_prior_scale", -6.9, -0.69),
    "seasonality_prior_scale": hp.loguniform("seasonality_prior_scale", -6.9, 2.3),
    "holidays_prior_scale": hp.loguniform("holidays_prior_scale", -6.9, 2.3),
    "seasonality_mode": hp.choice("seasonality_mode", seasonality_mode)
  }
  country_holidays = None
  run_parallel = True
 
  hyperopt_estim = ProphetHyperoptEstimator(horizon=horizon, frequency_unit=unit, metric="smape",interval_width=0.95,
                   country_holidays=country_holidays, search_space=search_space, num_folds=20, max_eval=10, trial_timeout=7108,
                   random_state=758501697, is_parallel=run_parallel)

  spark.conf.set("spark.databricks.mlflow.trackHyperopt.enabled", "false")

  results_pd = hyperopt_estim.fit(history_pd)

  spark.conf.set("spark.databricks.mlflow.trackHyperopt.enabled", "true")
 
  return results_pd[result_columns]

# COMMAND ----------

import mlflow
from databricks.automl_runtime.forecast.prophet.model import mlflow_prophet_log_model, ProphetModel

with mlflow.start_run(experiment_id="2255571809366092", run_name="Prophet") as mlflow_run:
  mlflow.set_tag("estimator_name", "Prophet")
  mlflow.log_param("interval_width", 0.95)
  df_aggregated = df_aggregated.rename(columns={time_col: "ds"})

  forecast_results = prophet_training(df_aggregated.to_pandas())
    
  # Log the metrics to mlflow
  metric_name_map = {"mse": "mean_squared_error", "rmse": "root_mean_squared_error", "mae": "mean_absolute_error",
                     "mape": "mean_absolute_percentage_error", "mdape": "mdape", "smape": "smape", "coverage": "coverage"}
  
  avg_metrics = forecast_results[metric_name_map.keys()].rename(columns=metric_name_map).mean().to_frame(name="mean_metrics").reset_index()
  avg_metrics["index"] = "val_" + avg_metrics["index"].astype(str)
  avg_metrics.set_index("index", inplace=True)
  mlflow.log_metrics(avg_metrics.to_dict()["mean_metrics"])
  
  # Create mlflow prophet model
  model_json = forecast_results["model_json"].to_list()[0]
  prophet_model = ProphetModel(model_json, horizon, unit, time_col)
  mlflow_prophet_log_model(prophet_model)

# COMMAND ----------

avg_metrics
# Uncomment to check the result details. By default, we only show the averaged metric becuase
# displaying the forecast result can take up a lot of storage for large datasets.
# forecast_results.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analyze the predicted results

# COMMAND ----------

# Load the model
run_id = mlflow_run.info.run_id
loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

# COMMAND ----------

future_df = loaded_model._model_impl.python_model.make_future_dataframe()
display(future_df.head(5))

# COMMAND ----------

# Predict future with the default horizon
forecast_pd = loaded_model._model_impl.python_model.model().predict(future_df)

# COMMAND ----------

# Plotly plots is turned off by default because it takes up a lot of storage.
# Set this flag to True and re-run the notebook to see the interactive plots with plotly
use_plotly = False

# COMMAND ----------

# Get prophet model
model = loaded_model._model_impl.python_model.model()
predict_pd = forecast_pd

# COMMAND ----------

# MAGIC %md
# MAGIC ### Plot the forecast with change points and trend
# MAGIC Plot the forecast using the `plot` method with your forecast dataframe. You can use `prophet.plot.add_changepoints_to_plot` to overlay significant changepoints. An interactive figure can be created with plotly.

# COMMAND ----------

from prophet.plot import add_changepoints_to_plot, plot_plotly

if use_plotly:
    fig = plot_plotly(model, predict_pd, changepoints=True, trend=True, figsize=(1200, 600))
else:
    fig = model.plot(predict_pd)
    a = add_changepoints_to_plot(fig.gca(), model, predict_pd)
fig

# COMMAND ----------

# MAGIC %md
# MAGIC ### Plot the forecast components
# MAGIC Use the `Prophet.plot_components` method to see the components. By default you'll see the trend, yearly seasonality, and weekly seasonality of the time series. You can also include holidays. An interactive figure can be created with plotly.

# COMMAND ----------

from prophet.plot import plot_components_plotly
if use_plotly:
    fig = plot_components_plotly(model, predict_pd, figsize=(900, 400))
    fig.show()
else:
    fig = model.plot_components(predict_pd)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Show the predicted results

# COMMAND ----------

predict_cols = ["ds", "yhat"]
forecast_pd = forecast_pd.reset_index()
display(forecast_pd[predict_cols].tail(horizon))