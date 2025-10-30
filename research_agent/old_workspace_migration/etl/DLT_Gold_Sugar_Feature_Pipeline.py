# Databricks notebook source
import dlt
import pyspark.sql.functions as F
from pyspark.sql.types import DateType

# --- 1. CONFIGURATION ---
TARGET_COMMODITY = "SUGAR"

# --- 2. DLT GOLD LAYER (Final Merge: Features + CFTC) ---
# This is the Gold table, ready for model training.

@dlt.table(
    name="gold_sugar_weekly_merged",
    comment="Final model-ready table combining engineered features and CFTC sentiment."
)
def sugar_weekly_merged():
    
    # 1. Read the upstream Silver features table using the fully qualified name
    # We must qualify the name because this pipeline will target the 'commodity.gold' schema.
    features_df = dlt.read("commodity.silver.silver_sugar_weekly")
    
    # 2. Read the raw CFTC data (Bronze Layer)
    cftc_df = dlt.read("commodity.bronze.cftc_data_raw")
    
    # Prepare CFTC Data
    cftc_df_sugar = cftc_df.filter(F.upper(F.col("market_and_exchange_names")) == TARGET_COMMODITY) 
    
    # Select and standardize CFTC columns
    cftc_df_sugar = cftc_df_sugar.select(
        F.col("as_of_date_in_form_yyyy-mm-dd").cast(DateType()).alias("Week_Start_Date"), 
        F.col("noncommercial_positions-long_all").alias("NC_Long"),
        F.col("noncommercial_positions-short_all").alias("NC_Short"),
        F.col("open_interest_all").alias("Open_Interest")
    ).filter(F.col("Week_Start_Date").isNotNull())

    # Calculate the Net Non-Commercial Position
    cftc_df_sugar = cftc_df_sugar.withColumn(
        "Net_Non_Commercial_Position",
        F.col("NC_Long") - F.col("NC_Short")
    )
    
    # 3. Perform Left Join to Merge
    final_merged_df = cftc_df_sugar.alias("cftc").join(
        features_df.alias("features"),
        on="Week_Start_Date",
        how="left" # Keep all CFTC weeks
    )

    # 4. Final Cleanup and Selection
    final_merged_df = final_merged_df.select(
        F.col("Week_Start_Date"),
        F.col("Net_Non_Commercial_Position"),  # TARGET VARIABLE
        F.col("Open_Interest"),
        # Select all features from the sugar_weekly table
        *[col for col in features_df.columns if col not in ("Week_Start_Date")] 
    ).na.drop(subset=["Net_Non_Commercial_Position"]) # Drop rows if we don't have the target

    return final_merged_df
