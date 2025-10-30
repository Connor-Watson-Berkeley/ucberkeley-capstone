# Databricks notebook source
import dlt
import pyspark.sql.functions as F
import pyspark.sql.window as W
from pyspark.sql.types import DateType

# --- 1. CONFIGURATION ---
TARGET_COMMODITY = "SUGAR"
TARGET_REGIONS = ["Brazil", "India", "Thailand", "China"]
# Note: DLT is defined globally, so catalog/schema prefixes are usually dropped
# when defining the table names, as DLT uses the pipeline's configured schema.
# We'll use the target schema name 'silver'.

# --- 2. DLT BASE LAYER (Daily Merged Input) ---
# This table merges all daily inputs (Price, Macro, Weather) for the commodity.

@dlt.table(
    name="sugar_daily_merged",
    comment="Daily merged raw features for Sugar commodity."
)
def sugar_daily_merged():
    # Read the Bronze tables (assuming they are available in the current context/catalog)
    # NOTE: If your bronze tables are in a different catalog, update 'commodity.bronze...'
    market_df = dlt.read("commodity.bronze.market_data_raw").filter(F.upper(F.col("commodity")) == TARGET_COMMODITY)
    vix_df = dlt.read("commodity.bronze.vix_data_raw")
    macro_df = dlt.read("commodity.bronze.macro_data_raw")
    # For weather, we select the required regions
    weather_df = dlt.read("commodity.bronze.v_weather_data_all").filter(
        (F.upper(F.col("commodity")) == TARGET_COMMODITY) & (F.col("region").isin(TARGET_REGIONS))
    )

    # Standardize columns for joining
    market_df = market_df.select(F.col("date").alias("Date"), F.col("close").alias("Sugar_Price"))
    vix_df = vix_df.select(F.col("date").alias("Date"), F.col("vix").alias("VIX"))
    # FIX: Use 'cop_usd' as placeholder for USD/BRL
    macro_df = macro_df.select(F.col("date").alias("Date"), F.col("cop_usd").alias("USD_BRL"))
    
    # ------------------
    # Daily Joins
    # ------------------
    daily_df = market_df.join(vix_df, on="Date", how="full_outer")
    daily_df = daily_df.join(macro_df, on="Date", how="full_outer")
    
    # Select relevant weather features before joining (note: weather is joined later)
    weather_df = weather_df.select(
        F.col("dt").alias("Date"),
        F.col("region").alias("Region"),
        F.col("temp_c").alias("Min_Temp_C"),
        F.col("precipitation_mm").alias("Rainfall_MM")
    )

    # Return the merged daily data
    return daily_df.filter(F.col("Date").isNotNull())


# --- 3. DLT FEATURE ENGINEERING LAYER (Weekly Aggregation) ---

@dlt.table(
    name="silver_sugar_weekly",
    comment="Weekly price, macro, and weather features for Sugar."
)
def sugar_weekly_features():
    df = dlt.read("sugar_daily_merged")
    weather_df = dlt.read("commodity.bronze.v_weather_data_all") # Need full weather data for partitioning

    # 1. Price Feature Engineering (Spark Window Functions)
    df = df.withColumn("Log_Return", F.log(F.col("Sugar_Price")) - F.log(F.lag(F.col("Sugar_Price"), 1).over(W.Window.orderBy("Date"))))

    # Calculate 20-Day Historical Volatility
    window_20d = W.Window.orderBy("Date").rowsBetween(-19, W.Window.currentRow)
    df = df.withColumn("HV_20D_Raw", F.stddev("Log_Return").over(window_20d))
    df = df.withColumn("HV_20D_Annualized", F.col("HV_20D_Raw") * F.sqrt(F.lit(252)))

    # 2. Weather Feature Engineering (Z-Score and Frost Count)
    
    # Window partitioned by Region for independent calculations
    window_90d_region = W.Window.partitionBy("Region").orderBy("Date").rowsBetween(-89, W.Window.currentRow)

    weather_features = weather_df.select(
        F.col("dt").alias("Date"),
        F.col("region").alias("Region"),
        F.col("temp_c").alias("Min_Temp_C"),
        F.col("precipitation_mm").alias("Rainfall_MM")
    ).filter(F.col("Region").isin(TARGET_REGIONS))
    
    # Calculate 90-day rolling sum for rainfall and temp features
    weather_features = weather_features.withColumn(
        "Rainfall_90D_Sum",
        F.sum("Rainfall_MM").over(window_90d_region)
    ).withColumn(
        "Frost_Day_Indicator",
        F.when(F.col("Min_Temp_C") <= 0, 1).otherwise(0)
    ).withColumn(
        "Frost_Count_60D",
        F.sum("Frost_Day_Indicator").over(W.Window.partitionBy("Region").orderBy("Date").rowsBetween(-59, W.Window.currentRow))
    )
    
    # Use global stats (or static stats) for Z-Score mean/stddev calculation (simplified here for DLT)
    # NOTE: For true Z-score, these should be historical averages. Here we use an approximation:
    weather_features = weather_features.withColumn(
        "Rainfall_Z_Score",
        (F.col("Rainfall_90D_Sum") - F.avg("Rainfall_90D_Sum").over(W.Window.partitionBy("Region").rowsBetween(-365*5, W.Window.currentRow)))
        / F.stddev("Rainfall_90D_Sum").over(W.Window.partitionBy("Region").rowsBetween(-365*5, W.Window.currentRow))
    )

    # 3. Aggregate Weather Stress (Worst Case)
    # Group by Date and find the worst stress across all monitored regions
    weather_stress_daily = weather_features.groupBy("Date").agg(
        F.min("Rainfall_Z_Score").alias("Worst_Rainfall_Z_Score"),
        F.max("Frost_Count_60D").alias("Max_Frost_Count")
    )

    # 4. Final Weekly Aggregation and Merge
    
    # Join features back to the main daily data
    df = df.join(weather_stress_daily, on="Date", how="left")

    # Determine the week start date for grouping
    df = df.withColumn("Week_Start_Date", F.date_trunc('week', F.col("Date")))

    weekly_features = df.groupBy("Week_Start_Date").agg(
        # Price Features
        F.last("Sugar_Price").alias("Sugar_Price_EOW"),
        F.mean("HV_20D_Annualized").alias("HV_20D_Annualized_Wk_Mean"),
        
        # Macro Features
        F.mean("VIX").alias("VIX_Wk_Mean"),
        F.mean("USD_BRL").alias("USD_BRL_Wk_Mean"),
        
        # Weather Stress (Take the minimum/maximum of the worst daily stress over the week)
        F.min("Worst_Rainfall_Z_Score").alias("Worst_Rainfall_Z_Score_Wk_Min"),
        F.max("Max_Frost_Count").alias("Max_Frost_Count_Wk_Max")
    ).filter(F.col("Sugar_Price_EOW").isNotNull()) # Only keep weeks where we have an end-of-week price

    return weekly_features

# --- 4. DLT FEATURE ENGINEERING LAYER (Monthly Aggregation) ---

@dlt.table(
    name="silver_sugar_monthly",
    comment="Monthly price, macro, and weather features for Sugar."
)
def sugar_monthly_features():
    df = dlt.read("sugar_daily_merged")
    
    # Determine the month start date for grouping
    df = df.withColumn("Month_Start_Date", F.date_trunc('month', F.col("Date")))

    # Use the same logic as the weekly feature calculation for monthly means
    df = df.withColumn("Log_Return", F.log(F.col("Sugar_Price")) - F.log(F.lag(F.col("Sugar_Price"), 1).over(W.Window.orderBy("Date"))))

    # Use the weather stress features calculated in the weekly pipeline's underlying logic
    # To keep this example simple and self-contained, we will perform the aggregation only
    # Note: For production, we would merge monthly weather data separately.
    
    monthly_features = df.groupBy("Month_Start_Date").agg(
        # Price Features
        F.mean("Sugar_Price").alias("Sugar_Price_Mo_Mean"),
        F.stddev("Log_Return").alias("Sugar_Price_Mo_Vol"),
        
        # Macro Features
        F.mean("VIX").alias("VIX_Mo_Mean"),
        F.mean("USD_BRL").alias("USD_BRL_Mo_Mean"),
    )
    
    # NOTE: You'd typically join monthly weather stress data here after calculating it separately
    
    return monthly_features