# Databricks notebook source
# MAGIC %md
# MAGIC # Data Exploration
# MAGIC - This notebook performs exploratory data analysis on the dataset.
# MAGIC - To expand on the analysis, attach this notebook to a cluster with runtime version **16.4.x-cpu-ml-scala2.12**, and rerun it.
# MAGIC - Explore completed trials in the [MLflow experiment](#mlflow/experiments/2255571809366092).

# COMMAND ----------

import os
import uuid
import pandas as pd
import shutil
import databricks.automl_runtime
import pyspark.pandas as ps

import mlflow

ps.options.plotting.backend = "matplotlib"

# Download input data from mlflow into a pyspark.pandas DataFrame
# create temp directory to download data
exp_temp_dir = os.path.join(os.environ["SPARK_LOCAL_DIRS"], "tmp", str(uuid.uuid4())[:8])
os.makedirs(exp_temp_dir)

# download the artifact and read it
exp_data_path = mlflow.artifacts.download_artifacts(run_id="3b626abacc5c4712bffdf63a64de53e7", artifact_path="data", dst_path=exp_temp_dir)
exp_file_path = os.path.join(exp_data_path, "training_data")
exp_file_path  = "file://" + exp_file_path

df = ps.from_pandas(pd.read_parquet(exp_file_path)).spark.cache()

target_col = "high"
time_col = "date"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Aggregate data

# COMMAND ----------

group_cols = [time_col]

df_aggregated = df \
  .groupby(group_cols) \
  .agg(high=(target_col, "avg")) \
  .reset_index()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Time column Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC Show the time range for the time series

# COMMAND ----------

df_time_range = df_aggregated[time_col].agg(["min", "max"])
df_time_range

# COMMAND ----------

# MAGIC %md
# MAGIC ## Target Value Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC Time series target value status

# COMMAND ----------

target_stats_df = df_aggregated[target_col].describe()
display(target_stats_df.reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC Check the number of missing values in the target column.

# COMMAND ----------

def num_nulls(x):
  num_nulls = x.isnull().sum()
  return pd.Series(num_nulls)

null_stats_df = df_aggregated.apply(num_nulls)[target_col]
null_stats_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualize the Data

# COMMAND ----------

df_sub = df_aggregated

df_sub = df_sub.filter(items=[time_col, target_col])
df_sub.set_index(time_col, inplace=True)
df_sub[target_col] = df_sub[target_col].astype("float")

# COMMAND ----------

import matplotlib.pyplot as plt

plt.plot(df_sub, label=target_col)
plt.legend()
plt.show()

# COMMAND ----------

# delete the temp data
shutil.rmtree(exp_temp_dir)