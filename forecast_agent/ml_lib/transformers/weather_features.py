"""
Weather feature transformers for unpacking weather_data arrays.

Provides aggregation strategies:
- Mean across all regions
- Weighted mean by production volume (TODO: add weights)
- Individual region features (expanded columns)
"""
from pyspark.ml.base import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCols, Param, Params
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, expr, avg as spark_avg
from typing import List


class WeatherAggregator(Transformer, HasInputCol):
    """
    Aggregate weather_data array into scalar features.

    Transforms ARRAY<STRUCT<region, temp_mean_c, ...>> into individual columns
    using mean aggregation across all regions.

    Input schema:
        weather_data: ARRAY<STRUCT<
            region: STRING,
            temp_max_c: DOUBLE,
            temp_min_c: DOUBLE,
            temp_mean_c: DOUBLE,
            precipitation_mm: DOUBLE,
            rain_mm: DOUBLE,
            snowfall_cm: DOUBLE,
            humidity_mean_pct: DOUBLE,
            wind_speed_max_kmh: DOUBLE
        >>

    Output columns (added to DataFrame):
        - weather_temp_mean_c: DOUBLE
        - weather_temp_max_c: DOUBLE
        - weather_temp_min_c: DOUBLE
        - weather_precipitation_mm: DOUBLE
        - weather_rain_mm: DOUBLE
        - weather_snowfall_cm: DOUBLE
        - weather_humidity_mean_pct: DOUBLE
        - weather_wind_speed_max_kmh: DOUBLE

    Example:
        transformer = WeatherAggregator(
            inputCol="weather_data",
            aggregation="mean"
        )
        df_transformed = transformer.transform(df)
    """

    aggregation = Param(
        Params._dummy(),
        "aggregation",
        "Aggregation strategy: 'mean' or 'weighted'"
    )

    def __init__(self, inputCol: str = "weather_data", aggregation: str = "mean"):
        super(WeatherAggregator, self).__init__()
        self._setDefault(aggregation="mean")
        self._set(inputCol=inputCol, aggregation=aggregation)

    def _transform(self, df: DataFrame) -> DataFrame:
        """
        Transform weather array into aggregate features.

        Uses aggregate() with named_struct to compute mean across all regions.
        """
        input_col = self.getInputCol()
        agg_type = self.getOrDefault(self.aggregation)

        if agg_type == "mean":
            # Aggregate array elements to compute mean across regions
            # Using aggregate() function: aggregate(array, initialValue, merge, finish)
            df = df.withColumn(
                "weather_temp_mean_c",
                expr(f"aggregate({input_col}, 0D, (acc, x) -> acc + x.temp_mean_c, acc -> acc / size({input_col}))")
            )
            df = df.withColumn(
                "weather_temp_max_c",
                expr(f"aggregate({input_col}, 0D, (acc, x) -> acc + x.temp_max_c, acc -> acc / size({input_col}))")
            )
            df = df.withColumn(
                "weather_temp_min_c",
                expr(f"aggregate({input_col}, 0D, (acc, x) -> acc + x.temp_min_c, acc -> acc / size({input_col}))")
            )
            df = df.withColumn(
                "weather_precipitation_mm",
                expr(f"aggregate({input_col}, 0D, (acc, x) -> acc + x.precipitation_mm, acc -> acc / size({input_col}))")
            )
            df = df.withColumn(
                "weather_rain_mm",
                expr(f"aggregate({input_col}, 0D, (acc, x) -> acc + x.rain_mm, acc -> acc / size({input_col}))")
            )
            df = df.withColumn(
                "weather_snowfall_cm",
                expr(f"aggregate({input_col}, 0D, (acc, x) -> acc + x.snowfall_cm, acc -> acc / size({input_col}))")
            )
            df = df.withColumn(
                "weather_humidity_mean_pct",
                expr(f"aggregate({input_col}, 0D, (acc, x) -> acc + x.humidity_mean_pct, acc -> acc / size({input_col}))")
            )
            df = df.withColumn(
                "weather_wind_speed_max_kmh",
                expr(f"aggregate({input_col}, 0D, (acc, x) -> acc + x.wind_speed_max_kmh, acc -> acc / size({input_col}))")
            )
        else:
            raise ValueError(f"Unsupported aggregation type: {agg_type}")

        return df

    def setInputCol(self, value: str):
        return self._set(inputCol=value)

    def setAggregation(self, value: str):
        return self._set(aggregation=value)
