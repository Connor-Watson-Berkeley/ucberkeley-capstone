# Databricks notebook source
# === SETUP: Upload Data ===
# Run this once to create folder and upload file
dbutils.fs.mkdirs("dbfs:/FileStore/gibbons_tony/commodity_data/")

# Then manually drag-and-drop prices_daily.csv here, or use:
# Upload your prices_daily.csv to /FileStore/tables/ via GUI
# Then run this to move it:
# dbutils.fs.cp("dbfs:/FileStore/tables/prices_daily.csv",
#               "dbfs:/FileStore/gibbons_tony/commodity_data/prices_daily.csv")