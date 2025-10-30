# Databricks notebook source

import dlt
from pyspark.sql.functions import (
    col, split, regexp_replace, explode, array_contains,
    to_date, substring, when, lower, lit, current_timestamp,
    avg, count, countDistinct, expr
)

# ---------------------------------------------------------------------------
# BRONZE
# ---------------------------------------------------------------------------

@dlt.table(
    name="commodity.bronze.bronze_gkg",
    comment="Raw GDELT GKG records from S3 (JSON Lines format)",
    table_properties={
        "quality": "bronze"
    }
)
def bronze_gkg():
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "json")
        .option("cloudFiles.schemaLocation", "s3://berkeley-datasci210-capstone/_schemas/gdelt")
        .option("cloudFiles.inferColumnTypes", "true")
        .option("cloudFiles.schemaEvolutionMode", "addNewColumns")
        .option("maxFilesPerTrigger", 100)
        .load("s3a://berkeley-datasci210-capstone/landing/gdelt/filtered/")
        .withColumn("ingestion_timestamp", current_timestamp())
    )

# ---------------------------------------------------------------------------
# SILVER (parsed)
# ---------------------------------------------------------------------------

@dlt.table(
    name="commodity.silver.silver_gkg",
    comment="Parsed and cleaned GDELT records with structured fields",
    table_properties={
        "quality": "silver",
        "delta.autoOptimize.optimizeWrite": "true",
        "delta.autoOptimize.autoCompact": "true"
    }
)
@dlt.expect_or_drop("valid_date", "date IS NOT NULL AND length(date) = 14")
@dlt.expect_or_drop("has_themes_or_names", "(themes_array IS NOT NULL AND size(themes_array) > 0) OR (all_names IS NOT NULL AND length(trim(all_names)) > 0)")
def silver_gkg():
    df = (
        dlt.read_stream("commodity.bronze.bronze_gkg")
        .withColumn("date_parsed", to_date(substring(col("date"), 1, 8), "yyyyMMdd"))
        .withColumn("date_partition", substring(col("date"), 1, 8))
        .withColumn("hour", substring(col("date"), 9, 2).cast("int"))
        .withColumn("minute", substring(col("date"), 11, 2).cast("int"))
        .withColumn("themes_clean", regexp_replace(col("themes"), r",\d+", ""))
        .withColumn("themes_array", split(col("themes_clean"), ";"))
        .withColumn("tone_split", split(col("tone"), ","))
        .withColumn("tone_score",        col("tone_split")[0].cast("double"))
        .withColumn("tone_positive",     col("tone_split")[1].cast("double"))
        .withColumn("tone_negative",     col("tone_split")[2].cast("double"))
        .withColumn("tone_polarity",     col("tone_split")[3].cast("double"))
        .withColumn("tone_activity_ref", col("tone_split")[4].cast("double"))
        .withColumn("tone_word_count",   col("tone_split")[6].cast("int"))
        .withColumn("locations_array", split(col("locations"), "#"))
        .withColumn("all_names_lower", lower(col("all_names")))
        .withColumn(
            "has_coffee",
            when(
                col("all_names_lower").contains("coffee") |
                col("all_names_lower").contains("arabica") |
                col("all_names_lower").contains("robusta"),
                lit(True)
            ).otherwise(lit(False))
        )
        .withColumn(
            "has_sugar",
            when(
                col("all_names_lower").contains("sugar") |
                col("all_names_lower").contains("sugarcane"),
                lit(True)
            ).otherwise(lit(False))
        )
        .select(
            "date", "date_parsed", "date_partition", "hour", "minute", "source_url",
            "themes_array", "locations_array", "all_names", "all_names_lower",
            "persons", "organizations",
            "tone_score", "tone_positive", "tone_negative", "tone_polarity",
            "tone_activity_ref", "tone_word_count",
            "has_coffee", "has_sugar", "ingestion_timestamp"
        )
    )
    return df

# ---------------------------------------------------------------------------
# SILVER (themes exploded)
# ---------------------------------------------------------------------------

@dlt.table(
    name="commodity.silver.silver_gkg_themes_exploded",
    comment="One row per theme per article for easier filtering and aggregation"
)
def silver_gkg_themes_exploded():
    return (
        dlt.read("silver_gkg")
        .select(
            "date", "date_parsed", "date_partition", "source_url",
            "tone_score", "has_coffee", "has_sugar",
            explode(col("themes_array")).alias("theme")
        )
        .filter(col("theme") != "")
    )

# ---------------------------------------------------------------------------
# GOLD: daily commodity sentiment
# ---------------------------------------------------------------------------

@dlt.table(
    name="commodity.gold.daily_commodity_sentiment",
    comment="Daily sentiment scores with volatility and dispersion metrics",
    table_properties={"quality": "gold"}
)
def gold_daily_commodity_sentiment():
    from pyspark.sql.window import Window
    from pyspark.sql.functions import stddev

    window_7d = Window.orderBy("date_parsed").rowsBetween(-6, 0)
    df = dlt.read("silver_gkg")

    daily_agg = (
        df.groupBy("date_parsed")
        .agg(
            count(when(col("has_coffee"), 1)).alias("coffee_article_count"),
            avg(when(col("has_coffee"), col("tone_score"))).alias("coffee_tone_mean"),
            stddev(when(col("has_coffee"), col("tone_score"))).alias("coffee_tone_disp"),
            avg(when(col("has_coffee"), col("tone_positive"))).alias("coffee_avg_positive"),
            avg(when(col("has_coffee"), col("tone_negative"))).alias("coffee_avg_negative"),

            count(when(col("has_sugar"), 1)).alias("sugar_article_count"),
            avg(when(col("has_sugar"), col("tone_score"))).alias("sugar_tone_mean"),
            stddev(when(col("has_sugar"), col("tone_score"))).alias("sugar_tone_disp"),
            avg(when(col("has_sugar"), col("tone_positive"))).alias("sugar_avg_positive"),
            avg(when(col("has_sugar"), col("tone_negative"))).alias("sugar_avg_negative"),

            count("*").alias("total_articles"),
            avg("tone_score").alias("tone_mean_global"),
            stddev("tone_score").alias("tone_disp_global")
        )
        .withColumn("coffee_tone_vol_7d", stddev("coffee_tone_mean").over(window_7d))
        .withColumn("sugar_tone_vol_7d",  stddev("sugar_tone_mean").over(window_7d))
        .withColumn("tone_vol_7d_global", stddev("tone_mean_global").over(window_7d))
    )
    return daily_agg.orderBy("date_parsed")

# ---------------------------------------------------------------------------
# GOLD: driver specific sentiment
# ---------------------------------------------------------------------------

@dlt.table(
    name="commodity.gold.driver_specific_sentiment",
    comment="Granular sentiment dynamics for specific drivers by region",
    table_properties={"quality": "gold"}
)
def gold_driver_specific_sentiment():
    from pyspark.sql.window import Window
    from pyspark.sql.functions import stddev

    window_7d = Window.orderBy("date_parsed").rowsBetween(-6, 0)
    df = dlt.read("silver_gkg")

    def create_filter(themes_regex, location_code=None):
        theme_filter = col("themes_array").cast("string").rlike(themes_regex)
        if location_code:
            location_filter = col("locations_array").cast("string").contains(location_code)
            return theme_filter & location_filter
        return theme_filter

    daily_drivers = (
        df.groupBy("date_parsed")
        .agg(
            count(when(create_filter("NATURAL_DISASTER|CLIMATE_CHANGE", "BR"), 1)).alias("weather_br_count"),
            avg(when(create_filter("NATURAL_DISASTER|CLIMATE_CHANGE", "BR"), col("tone_score"))).alias("tone_mean_weather_br"),
            stddev(when(create_filter("NATURAL_DISASTER|CLIMATE_CHANGE", "BR"), col("tone_score"))).alias("tone_disp_weather_br"),

            count(when(create_filter("NATURAL_DISASTER|CLIMATE_CHANGE", "VN"), 1)).alias("weather_vn_count"),
            avg(when(create_filter("NATURAL_DISASTER|CLIMATE_CHANGE", "VN"), col("tone_score"))).alias("tone_mean_weather_vn"),
            stddev(when(create_filter("NATURAL_DISASTER|CLIMATE_CHANGE", "VN"), col("tone_score"))).alias("tone_disp_weather_vn"),

            count(when(create_filter("NATURAL_DISASTER|CLIMATE_CHANGE", "IN"), 1)).alias("weather_in_count"),
            avg(when(create_filter("NATURAL_DISASTER|CLIMATE_CHANGE", "IN"), col("tone_score"))).alias("tone_mean_weather_in"),
            stddev(when(create_filter("NATURAL_DISASTER|CLIMATE_CHANGE", "IN"), col("tone_score"))).alias("tone_disp_weather_in"),

            count(when(create_filter("ECON_TRADE_DISPUTE|TAX_TARIFFS"), 1)).alias("trade_count"),
            avg(when(create_filter("ECON_TRADE_DISPUTE|TAX_TARIFFS"), col("tone_score"))).alias("tone_mean_trade"),
            stddev(when(create_filter("ECON_TRADE_DISPUTE|TAX_TARIFFS"), col("tone_score"))).alias("tone_disp_trade"),

            count(when(create_filter("ENERGY|OIL"), 1)).alias("energy_count"),
            avg(when(create_filter("ENERGY|OIL"), col("tone_score"))).alias("tone_mean_energy"),
            stddev(when(create_filter("ENERGY|OIL"), col("tone_score"))).alias("tone_disp_energy")
        )
        .withColumn("tone_vol_7d_weather_br", stddev("tone_mean_weather_br").over(window_7d))
        .withColumn("tone_vol_7d_weather_vn", stddev("tone_mean_weather_vn").over(window_7d))
        .withColumn("tone_vol_7d_weather_in", stddev("tone_mean_weather_in").over(window_7d))
        .withColumn("tone_vol_7d_trade",      stddev("tone_mean_trade").over(window_7d))
        .withColumn("tone_vol_7d_energy",     stddev("tone_mean_energy").over(window_7d))
    )
    return daily_drivers.orderBy("date_parsed")

# ---------------------------------------------------------------------------
# GOLD: daily theme counts
# ---------------------------------------------------------------------------

@dlt.table(
    name="commodity.gold.daily_theme_counts",
    comment="Daily counts of key themes relevant to commodity prices",
    table_properties={"quality": "gold"}
)
def gold_daily_theme_counts():
    key_themes = [
        'AGRICULTURE','FOOD_STAPLE','FOOD_SECURITY',
        'NATURAL_DISASTER','CLIMATE_CHANGE','DROUGHT',
        'TAX_DISEASE','TAX_PLANTDISEASE','TAX_PESTS',
        'ECON_SUBSIDIES','STRIKE','CRISIS_LOGISTICS',
        'ECON_TRADE_DISPUTE','TAX_TARIFFS','ELECTION',
        'ECON_INTEREST_RATES','ECON_CURRENCY_EXCHANGE_RATE'
    ]
    df = dlt.read("silver_gkg_themes_exploded")
    for theme in key_themes:
        df = df.withColumn(f"theme_{theme.lower()}", when(col("theme") == theme, 1).otherwise(0))
    agg_cols = [count(when(col(f"theme_{t.lower()}") == 1, 1)).alias(f"count_{t.lower()}") for t in key_themes]
    return (
        df.groupBy("date_parsed")
          .agg(*agg_cols, count("*").alias("total_theme_mentions"))
          .orderBy("date_parsed")
    )

# ---------------------------------------------------------------------------
# GOLD: coffee regions
# ---------------------------------------------------------------------------
def _country_from_locations(col_locations):
    # Uses a higher-order function: exists(array, x -> predicate(x))
    return (
        when(expr("exists(locations_array, x -> x LIKE '%#BR#%')"), "Brazil")
        .when(expr("exists(locations_array, x -> x LIKE '%#VN#%')"), "Vietnam")
        .when(expr("exists(locations_array, x -> x LIKE '%#CO#%')"), "Colombia")
        .when(expr("exists(locations_array, x -> x LIKE '%#ET#%')"), "Ethiopia")
        .when(expr("exists(locations_array, x -> x LIKE '%#HN#%')"), "Honduras")
        .when(expr("exists(locations_array, x -> x LIKE '%#ID#%')"), "Indonesia")
        .when(expr("exists(locations_array, x -> x LIKE '%#PE#%')"), "Peru")
        .otherwise("Other")
    )

@dlt.table(
    name="commodity.gold.coffee_production_regions",
    comment="Coffee-related articles by major production regions",
    table_properties={"quality": "gold"}
)
def gold_coffee_production_regions():
    df = dlt.read("silver_gkg").filter(col("has_coffee"))
    df = df.withColumn("country", _country_from_locations(col("locations_array")))

    return (
        df.groupBy("date_parsed", "country")
              .agg(count("*").alias("article_count"),
                   avg("tone_score").alias("avg_tone"),
                   avg("tone_negative").alias("avg_negative"),
                   countDistinct("source_url").alias("unique_sources"))
              .orderBy("date_parsed", "country")
    )

# ---------------------------------------------------------------------------
# GOLD: sugar regions
# ---------------------------------------------------------------------------

@dlt.table(
    name="commodity.gold.sugar_production_regions",
    comment="Sugar-related articles by major production regions",
    table_properties={"quality": "gold"}
)
def gold_sugar_production_regions():
    df = dlt.read("silver_gkg").filter(col("has_sugar"))
    df = df.withColumn(
        "country",
        when(expr("exists(locations_array, x -> x LIKE '%#BR#%')"), "Brazil")
        .when(expr("exists(locations_array, x -> x LIKE '%#IN#%')"), "India")
        .when(expr("exists(locations_array, x -> x LIKE '%#TH#%')"), "Thailand")
        .when(expr("exists(locations_array, x -> x LIKE '%#CN#%')"), "China")
        .when(expr("exists(locations_array, x -> x LIKE '%#US#%')"), "USA")
        .when(expr("exists(locations_array, x -> x LIKE '%#MX#%')"), "Mexico")
        .when(expr("exists(locations_array, x -> x LIKE '%#PK#%')"), "Pakistan")
        .otherwise("Other")
    )

    return (
        df.groupBy("date_parsed", "country")
              .agg(count("*").alias("article_count"),
                   avg("tone_score").alias("avg_tone"),
                   avg("tone_negative").alias("avg_negative"),
                   countDistinct("source_url").alias("unique_sources"))
              .orderBy("date_parsed", "country")
    )

# ---------------------------------------------------------------------------
# GOLD: theme co-occurrence
# ---------------------------------------------------------------------------

@dlt.table(
    name="commodity.gold.theme_cooccurrence",
    comment="Co-occurrence of themes within articles (e.g., AGRICULTURE + DROUGHT)",
    table_properties={"quality": "gold"}
)
def gold_theme_cooccurrence():
    return (
        dlt.read("silver_gkg")
        .select("date_parsed","source_url","themes_array","has_coffee","has_sugar","tone_score")
        .withColumn("has_agriculture",  array_contains(col("themes_array"), "AGRICULTURE"))
        .withColumn("has_disaster",     array_contains(col("themes_array"), "NATURAL_DISASTER"))
        .withColumn("has_climate",      array_contains(col("themes_array"), "CLIMATE_CHANGE"))
        .withColumn("has_trade_dispute",array_contains(col("themes_array"), "ECON_TRADE_DISPUTE"))
        .withColumn("has_tariffs",      array_contains(col("themes_array"), "TAX_TARIFFS"))
        .groupBy("date_parsed")
        .agg(
            count(when(col("has_agriculture") & col("has_disaster"), 1)).alias("agri_disaster_count"),
            count(when(col("has_agriculture") & col("has_climate"), 1)).alias("agri_climate_count"),
            count(when(col("has_agriculture") & col("has_trade_dispute"), 1)).alias("agri_trade_count"),
            count(when(col("has_coffee") & col("has_disaster"), 1)).alias("coffee_disaster_count"),
            count(when(col("has_coffee") & col("has_climate"), 1)).alias("coffee_climate_count"),
            count(when(col("has_sugar") & col("has_disaster"), 1)).alias("sugar_disaster_count"),
            count(when(col("has_sugar") & col("has_climate"), 1)).alias("sugar_climate_count")
        )
        .orderBy("date_parsed")
    )

# ---------------------------------------------------------------------------
# GOLD: weekly rolling sentiment
# ---------------------------------------------------------------------------

@dlt.table(
    name="commodity.gold.weekly_rolling_sentiment",
    comment="7-day rolling average sentiment for smoothing daily volatility",
    table_properties={"quality": "gold"}
)
def gold_weekly_rolling_sentiment():
    from pyspark.sql.window import Window
    window_spec = Window.orderBy("date_parsed").rowsBetween(-6, 0)

    return (
        dlt.read("commodity.gold.daily_commodity_sentiment")
        .withColumn("coffee_tone_7d_avg",  avg("coffee_tone_mean").over(window_spec))
        .withColumn("sugar_tone_7d_avg",   avg("sugar_tone_mean").over(window_spec))
        .withColumn("coffee_article_7d_avg", avg("coffee_article_count").over(window_spec))
        .withColumn("sugar_article_7d_avg",  avg("sugar_article_count").over(window_spec))
        .select(
            "date_parsed",
            "coffee_article_count",
            "coffee_tone_mean",
            "coffee_tone_7d_avg",
            "coffee_article_7d_avg",
            "sugar_article_count",
            "sugar_tone_mean",
            "sugar_tone_7d_avg",
            "sugar_article_7d_avg"
        )
    )

# ---------------------------------------------------------------------------
# GOLD: ML features
# ---------------------------------------------------------------------------

@dlt.table(
    name="commodity.gold.ml_features_daily",
    comment="Complete daily feature set with all sentiment dynamics for time series models",
    table_properties={"quality": "gold"}
)
def gold_ml_features_daily():
    sentiment_df = dlt.read("commodity.gold.daily_commodity_sentiment")
    themes_df    = dlt.read("commodity.gold.daily_theme_counts")
    cooccur_df   = dlt.read("commodity.gold.theme_cooccurrence")
    drivers_df   = dlt.read("commodity.gold.driver_specific_sentiment")

    return (
        sentiment_df
        .join(themes_df,  "date_parsed", "left")
        .join(cooccur_df, "date_parsed", "left")
        .join(drivers_df, "date_parsed", "left")
        .fillna(0)
        .orderBy("date_parsed")
    )
