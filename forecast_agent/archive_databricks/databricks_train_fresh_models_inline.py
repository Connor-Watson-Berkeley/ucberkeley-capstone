# Databricks notebook source
"""
Train Fresh Models in Databricks (Feature Engineering Inlined)
Trains Coffee & Sugar models using Databricks default package versions
"""

# COMMAND ----------

# MAGIC %pip install databricks-sql-connector xgboost statsmodels pmdarima

# COMMAND ----------

# Restart Python to load newly installed packages
dbutils.library.restartPython()

# COMMAND ----------

print("=" * 80)
print("Training Fresh Models in Databricks")
print("=" * 80)
print("\nThis ensures NumPy/sklearn/xgboost version consistency")
print("All models will be trained using Databricks default package versions\n")

# COMMAND ----------

# Verify package versions (Databricks default)
import numpy as np
import sklearn
import xgboost as xgb
import pandas as pd
import sys
import os
from datetime import datetime, timedelta

print(f"\n📦 Package Versions (Databricks):")
print(f"  NumPy: {np.__version__}")
print(f"  scikit-learn: {sklearn.__version__}")
print(f"  XGBoost: {xgb.__version__}")
print(f"  Pandas: {pd.__version__}")

# COMMAND ----------

# INLINE FEATURE ENGINEERING FUNCTIONS
# (Copied from ground_truth/features/data_preparation.py to avoid import issues)

def prepare_data_for_model(
    raw_data: pd.DataFrame,
    commodity: str,
    region_strategy: str = 'aggregate',
    gdelt_strategy = None,
    gdelt_themes = None,
    feature_columns = None
) -> pd.DataFrame:
    """
    Prepare data for model training/prediction with flexible aggregation.

    Args:
        raw_data: Raw data from unified_data (may have multiple regions per date)
        commodity: Commodity name ('Coffee' or 'Sugar')
        region_strategy: How to handle region data
            - 'aggregate': Average across regions
            - 'pivot': Create region-specific columns
            - 'all': Keep multi-row format (for neural models)
        gdelt_strategy: How to handle GDELT theme data (if None, uses region_strategy)
        gdelt_themes: List of GDELT themes to use when gdelt_strategy='select'
        feature_columns: Optional list of specific feature columns to include

    Returns:
        Processed DataFrame ready for model training/prediction
    """
    if gdelt_strategy is None:
        gdelt_strategy = region_strategy

    # Filter by commodity
    df = raw_data[raw_data['commodity'] == commodity].copy()

    # Strategy: 'all' - Keep raw multi-row format
    if region_strategy == 'all':
        if feature_columns:
            return df[['date', 'commodity', 'region'] + feature_columns].set_index('date')
        return df.set_index('date')

    # Strategy: 'aggregate' - Average across regions
    elif region_strategy == 'aggregate':
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['commodity', 'region']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

        agg_dict = {col: 'mean' for col in numeric_cols}
        df_agg = df.groupby('date').agg(agg_dict).reset_index()
        df_agg['commodity'] = commodity

        # Handle GDELT themes
        if gdelt_strategy == 'aggregate':
            df_agg = _aggregate_gdelt_themes(df_agg)
        elif gdelt_strategy == 'select' and gdelt_themes:
            df_agg = _select_gdelt_themes(df_agg, gdelt_themes)

        if feature_columns:
            keep_cols = ['date'] + [col for col in feature_columns if col in df_agg.columns]
            df_agg = df_agg[keep_cols]

        return df_agg.set_index('date')

    # Strategy: 'pivot' - Create region-specific columns
    elif region_strategy == 'pivot':
        df_pivot = _pivot_regions(df, feature_columns)

        if gdelt_strategy == 'pivot':
            pass
        elif gdelt_strategy == 'select' and gdelt_themes:
            df_pivot = _select_gdelt_themes(df_pivot, gdelt_themes)
        elif gdelt_strategy == 'aggregate':
            df_pivot = _aggregate_gdelt_themes(df_pivot)

        return df_pivot.set_index('date')

    else:
        raise ValueError(f"Invalid region_strategy: {region_strategy}")


def _pivot_regions(df: pd.DataFrame, feature_columns = None) -> pd.DataFrame:
    """Pivot region data to create region-specific columns."""
    if 'region' not in df.columns:
        return df

    if feature_columns:
        pivot_cols = [col for col in feature_columns if col in df.columns and col not in ['date', 'commodity', 'region', 'close']]
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        pivot_cols = [col for col in numeric_cols if col not in ['close']]

    pivot_cols_with_target = pivot_cols + ['close']
    regions = df['region'].unique()

    dfs = []
    for region in regions:
        region_df = df[df['region'] == region].copy()
        region_name = region.lower().replace(' ', '_')
        rename_dict = {col: f"{region_name}_{col}" for col in pivot_cols_with_target}
        region_df = region_df.rename(columns=rename_dict)
        keep_cols = ['date'] + list(rename_dict.values())
        region_df = region_df[keep_cols]
        dfs.append(region_df)

    df_pivot = dfs[0]
    for region_df in dfs[1:]:
        df_pivot = df_pivot.merge(region_df, on='date', how='outer')

    return df_pivot


def _aggregate_gdelt_themes(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate GDELT themes by averaging across theme-specific columns."""
    gdelt_cols = [col for col in df.columns if col.startswith('group_')]

    if not gdelt_cols:
        return df

    metric_groups = {}
    for col in gdelt_cols:
        parts = col.split('_')
        if len(parts) >= 3:
            metric = '_'.join(parts[2:])
            if metric not in metric_groups:
                metric_groups[metric] = []
            metric_groups[metric].append(col)

    for metric, cols in metric_groups.items():
        df[f'gdelt_{metric}_all'] = df[cols].mean(axis=1)

    df = df.drop(columns=gdelt_cols)
    return df


def _select_gdelt_themes(df: pd.DataFrame, themes) -> pd.DataFrame:
    """Keep only specific GDELT themes."""
    keep_cols = []
    for col in df.columns:
        if not col.startswith('group_'):
            keep_cols.append(col)
        else:
            for theme in themes:
                if f'group_{theme}_' in col:
                    keep_cols.append(col)
                    break

    return df[keep_cols]


print("✓ Feature engineering functions defined (inline)")

# COMMAND ----------

# Set up paths for imports from Git repo
forecast_agent_path = '/Workspace/Repos/Project_Git/ucberkeley-capstone/forecast_agent'
if forecast_agent_path not in sys.path:
    sys.path.insert(0, forecast_agent_path)

print(f"✓ Added {forecast_agent_path} to Python path")

# Change to forecast_agent directory for relative imports
os.chdir(forecast_agent_path)
print(f"✓ Changed directory to {forecast_agent_path}")

# COMMAND ----------

# Import training modules from Git repo (only the ones we need, not data_preparation)
print("\n📥 Importing training modules from Git repo...")

from ground_truth.config.model_registry import BASELINE_MODELS
from train_models import get_training_dates

# Import model training functions (at top level, not in loop!)
from ground_truth.models import naive, random_walk, arima, sarimax, xgboost_model

print("✓ All modules imported successfully")
print("✓ Running in Databricks - using spark.sql() (no credentials needed!)")

# COMMAND ----------

# Configuration
commodities = ['Coffee', 'Sugar']
model_keys = ['naive', 'xgboost', 'sarimax_auto_weather']
train_frequency = 'semiannually'
start_date = datetime.strptime('2018-01-01', '%Y-%m-%d').date()
end_date = datetime.strptime('2025-11-17', '%Y-%m-%d').date()
model_version = 'v1.0'
min_training_days = 1095  # 3 years

print("\n" + "=" * 80)
print("TRAINING CONFIGURATION")
print("=" * 80)
print(f"Commodities: {', '.join(commodities)}")
print(f"Models: {', '.join(model_keys)}")
print(f"Training Frequency: {train_frequency}")
print(f"Date Range: {start_date} to {end_date}")
print(f"Model Version: {model_version}")
print(f"Min Training Days: {min_training_days}")
print("=" * 80)

# COMMAND ----------

# Helper functions for Spark-based data access
def load_training_data_spark(commodity: str, cutoff_date, lookback_days=1460, return_spark_df=False):
    """Load RAW training data using Spark SQL (multi-region format)

    Args:
        commodity: Commodity name
        cutoff_date: Latest date to include
        lookback_days: Days of history to load (default 1460 = 4 years)
        return_spark_df: If True, return cached Spark DF; if False, convert to Pandas

    Returns:
        Cached Spark DataFrame or Pandas DataFrame
    """
    from datetime import datetime, timedelta

    # Calculate lookback date
    if isinstance(cutoff_date, str):
        cutoff_dt = pd.to_datetime(cutoff_date)
    else:
        cutoff_dt = cutoff_date

    if lookback_days:
        lookback_date = cutoff_dt - timedelta(days=lookback_days)
        lookback_str = lookback_date.strftime('%Y-%m-%d')
        date_filter = f"AND date >= '{lookback_str}' AND date <= '{cutoff_date}'"
    else:
        date_filter = f"AND date <= '{cutoff_date}'"

    # Select only needed columns (not SELECT *)
    query = f"""
        SELECT
            date,
            commodity,
            region,
            close,
            temp_mean_c,
            humidity_mean_pct,
            precipitation_mm,
            vix
        FROM commodity.silver.unified_data
        WHERE commodity = '{commodity}'
        {date_filter}
    """

    # Execute query and cache
    spark_df = spark.sql(query).cache()

    # Return Spark DF for caching, or convert to Pandas
    if return_spark_df:
        return spark_df
    else:
        # Convert to Pandas and sort (sort in Pandas, not Spark)
        pandas_df = spark_df.toPandas()
        pandas_df['date'] = pd.to_datetime(pandas_df['date'])
        pandas_df = pandas_df.sort_values('date')

        return pandas_df


def prepare_training_data(commodity: str, cutoff_date, model_config: dict, cached_spark_df=None):
    """Load and prepare training data with feature engineering based on model config

    Args:
        commodity: Commodity name
        cutoff_date: Training cutoff date
        model_config: Model configuration dict
        cached_spark_df: Optional cached Spark DataFrame (for performance optimization)

    Returns:
        Prepared Pandas DataFrame with feature engineering applied
    """
    # Load raw multi-region data (use cached if provided, otherwise query)
    if cached_spark_df is not None:
        # Convert cached Spark DF to Pandas
        raw_df = cached_spark_df.toPandas()
        raw_df['date'] = pd.to_datetime(raw_df['date'])
        raw_df = raw_df.sort_values('date')
    else:
        # Load from Spark SQL
        raw_df = load_training_data_spark(commodity, cutoff_date)

    # Get model parameters for feature engineering
    params = model_config.get('params', {})
    region_strategy = params.get('region_strategy', 'aggregate')
    gdelt_strategy = params.get('gdelt_strategy', None)
    gdelt_themes = params.get('gdelt_themes', None)
    feature_columns = params.get('exog_features', None)

    # If feature_columns is specified, add target column
    if feature_columns:
        target = params.get('target', 'close')
        feature_columns_with_target = [target] + feature_columns
    else:
        feature_columns_with_target = None

    # Apply feature engineering using our inline function
    prepared_df = prepare_data_for_model(
        raw_data=raw_df,
        commodity=commodity,
        region_strategy=region_strategy,
        gdelt_strategy=gdelt_strategy,
        gdelt_themes=gdelt_themes,
        feature_columns=feature_columns_with_target
    )

    return prepared_df


def model_exists_spark(commodity: str, model_name: str, training_cutoff: str, model_version: str) -> bool:
    """Check if model already exists using Spark SQL"""
    query = f"""
        SELECT COUNT(*) as count
        FROM commodity.forecast.trained_models
        WHERE commodity = '{commodity}'
        AND model_name = '{model_name}'
        AND training_cutoff_date = '{training_cutoff}'
        AND model_version = '{model_version}'
    """
    result = spark.sql(query).collect()[0]
    return result['count'] > 0


def save_model_spark(fitted_model_dict: dict, commodity: str, model_name: str, model_version: str, created_by: str):
    """Save trained model using Spark SQL"""
    import json
    from datetime import datetime

    # Extract training date range
    training_start_date = fitted_model_dict['first_date'].strftime('%Y-%m-%d')
    training_cutoff_date = fitted_model_dict['last_date'].strftime('%Y-%m-%d')

    # Serialize model to JSON
    model_json = json.dumps(fitted_model_dict, default=str)
    model_size_mb = len(model_json) / (1024 * 1024)

    print(f"      💾 Model size: {model_size_mb:.2f} MB")
    print(f"      📅 Training range: {training_start_date} to {training_cutoff_date}")

    if model_size_mb >= 1.0:
        print(f"      ⚠️  Model too large for JSON ({model_size_mb:.2f} MB ≥ 1 MB) - skipping S3 upload for now")
        return None

    # Create record - partition by cutoff date
    year = int(training_cutoff_date[:4])
    month = int(training_cutoff_date[5:7])

    # Use Spark SQL to insert
    insert_query = f"""
        INSERT INTO commodity.forecast.trained_models
        (commodity, model_name, model_version, training_start_date, training_cutoff_date, fitted_model_json,
         created_at, created_by, year, month)
        VALUES (
            '{commodity}',
            '{model_name}',
            '{model_version}',
            '{training_start_date}',
            '{training_cutoff_date}',
            '{model_json.replace("'", "''")}',
            '{datetime.utcnow().isoformat()}',
            '{created_by}',
            {year},
            {month}
        )
    """

    spark.sql(insert_query)
    print(f"      ✅ Model saved to trained_models table")
    return f"{commodity}_{model_name}_{training_cutoff_date}_{model_version}"


print("✓ Spark helper functions defined")

# COMMAND ----------

# Train models for each commodity
for commodity in commodities:
    print("\n" + "=" * 80)
    print(f"TRAINING {commodity.upper()} MODELS")
    print("=" * 80)

    # Generate training dates
    training_dates = get_training_dates(start_date, end_date, train_frequency)
    print(f"\n📅 Training Windows: {len(training_dates)} windows from {training_dates[0]} to {training_dates[-1]}")

    total_trained = 0
    total_skipped = 0
    total_failed = 0

    # Training loop
    for window_idx, training_cutoff in enumerate(training_dates, 1):
        print(f"\n{'='*80}")
        print(f"Window {window_idx}/{len(training_dates)}: Training Cutoff = {training_cutoff}")
        print(f"{'='*80}")

        # OPTIMIZATION: Load data ONCE per window and cache it (instead of per-model)
        # This reduces 96+ queries to 32 queries (16 windows × 2 commodities)
        print(f"\n📥 Loading data for window (cached, 4-year lookback)...")
        cached_spark_df = load_training_data_spark(commodity, training_cutoff, return_spark_df=True)
        print(f"   ✓ Data cached in memory - will reuse for all {len(model_keys)} models")

        # Train each model with model-specific feature engineering
        for model_key in model_keys:
            model_config = BASELINE_MODELS[model_key]
            model_name = model_config['name']

            print(f"\n   🔧 {model_name} ({model_key}):")

            # Check if model already exists
            if model_exists_spark(commodity, model_name, training_cutoff.strftime('%Y-%m-%d'), model_version):
                print(f"      ⏩ Model already exists - skipping")
                total_skipped += 1
                continue

            # Use cached data for feature engineering (no Spark query needed!)
            print(f"      🔄 Preparing features from cached data...")
            training_df = prepare_training_data(commodity, training_cutoff, model_config, cached_spark_df=cached_spark_df)

            # Check minimum training days
            if len(training_df) < min_training_days:
                print(f"      ⚠️  Insufficient training data: {len(training_df)} days < {min_training_days} days - skipping")
                total_skipped += 1
                continue

            print(f"      📊 Loaded {len(training_df):,} days of training data")
            print(f"      📅 Data range: {training_df.index[0].date()} to {training_df.index[-1].date()}")

            try:
                # Train the model
                from ground_truth.models import naive, random_walk, arima, sarimax, xgboost_model

                # Map model keys to training functions
                train_functions = {
                    'naive': naive.naive_train,
                    'random_walk': random_walk.random_walk_train,
                    'arima_111': arima.arima_train,
                    'sarimax_auto': sarimax.sarimax_train,
                    'sarimax_auto_weather': sarimax.sarimax_train,
                    'xgboost': xgboost_model.xgboost_train,
                }

                train_func = train_functions.get(model_key)
                if not train_func:
                    print(f"      ❌ No training function for {model_key}")
                    total_failed += 1
                    continue

                # Get model parameters
                params = model_config['params'].copy()

                # Remove parameters that are not used during training (only used during prediction)
                train_params = {k: v for k, v in params.items() if k not in ['horizon', 'region_strategy', 'gdelt_strategy', 'gdelt_themes']}

                # Train model
                fitted_model_dict = train_func(training_df, **train_params)

                # Ensure first_date is tracked (for training range visibility)
                if 'first_date' not in fitted_model_dict:
                    fitted_model_dict['first_date'] = training_df.index[0]

                # Save model
                model_id = save_model_spark(
                    fitted_model_dict=fitted_model_dict,
                    commodity=commodity,
                    model_name=model_name,
                    model_version=model_version,
                    created_by="databricks_train_fresh_models_inline.py"
                )

                if model_id:
                    total_trained += 1
                else:
                    total_failed += 1

            except Exception as e:
                print(f"      ❌ Training failed: {str(e)[:200]}")
                total_failed += 1

    # Summary for this commodity
    print("\n" + "=" * 80)
    print(f"{commodity.upper()} TRAINING COMPLETE")
    print("=" * 80)
    print(f"✅ Models Trained: {total_trained}")
    print(f"⏩ Models Skipped (already exist): {total_skipped}")
    print(f"❌ Models Failed: {total_failed}")
    print("=" * 80)

# COMMAND ----------

# Verify trained models in database
print("\n" + "=" * 80)
print("VERIFYING TRAINED MODELS IN DATABASE")
print("=" * 80)

for commodity in commodities:
    print(f"\n📊 {commodity} Models:")
    result = spark.sql(f"""
        SELECT model_name, COUNT(*) as count
        FROM commodity.forecast.trained_models
        WHERE commodity = '{commodity}'
        AND model_name IN ('Naive', 'XGBoost', 'SARIMAX+Weather')
        GROUP BY model_name
        ORDER BY model_name
    """).collect()

    for row in result:
        print(f"  {row['model_name']}: {row['count']} models")

# COMMAND ----------

print("\n" + "=" * 80)
print("✅ ALL TRAINING COMPLETE!")
print("=" * 80)
print("\n📝 Summary:")
print("  ✓ All locally-trained models deleted (from previous session)")
print("  ✓ Fresh models trained in Databricks")
print("  ✓ Using Databricks default package versions")
print("  ✓ Version consistency guaranteed")
print("  ✓ Feature engineering inlined (no Git repo dependencies)")
print("\nNext: Run backfill_rolling_window_spark.py in same environment")
