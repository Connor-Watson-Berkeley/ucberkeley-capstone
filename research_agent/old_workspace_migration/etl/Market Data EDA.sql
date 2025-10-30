-- Databricks notebook source
select * from workspace_ingestion_data.default.market_data;

-- COMMAND ----------

-- MAGIC %md ## 1. dddddd

-- COMMAND ----------

-- MAGIC %python
-- MAGIC spark.getActiveSession().sql("set spark.sql.legacy.allowUntypedScalaUDF=true")

-- COMMAND ----------

-- MAGIC %python import pandas as pd
-- MAGIC         import numpy as np
-- MAGIC         import matplotlib.pyplot as plt
-- MAGIC         import seaborn as sns
-- MAGIC         import warnings
-- MAGIC
-- MAGIC         df = spark.sql('''''')
-- MAGIC         df

-- COMMAND ----------

SELECT * FROM workspace_ingestion_data.default.cftc;