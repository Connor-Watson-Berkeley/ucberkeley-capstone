"""
Multi-Regional Pivot Experiment

Tests whether using ALL 22 regions' weather as separate features (pivoted)
improves forecast accuracy compared to single-region or aggregated approaches.

Feature structure:
- weather_bahia_brazil_temp_max
- weather_vietnam_temp_max
- weather_colombia_temp_max
- ... (22 regions × 8 weather features = 176 features)
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from darts import TimeSeries
from darts.models import NHiTSModel, NBEATSModel, TCNModel, TransformerModel
from darts.metrics import mape, rmse, mae
import warnings
warnings.filterwarnings('ignore')


def download_fresh_data():
    """Download unified_data with all regions and weather columns."""
    print("=" * 80)
    print("DOWNLOADING FRESH DATA")
    print("=" * 80)

    from databricks import sql

    conn = sql.connect(
        server_hostname=os.environ['DATABRICKS_HOST'],
        http_path=os.environ['DATABRICKS_HTTP_PATH'],
        access_token=os.environ['DATABRICKS_TOKEN']
    )

    cursor = conn.cursor()
    query = """
        SELECT *
        FROM commodity.silver.unified_data
        WHERE commodity = 'Coffee'
        ORDER BY date, region
    """

    print("Querying Databricks...")
    cursor.execute(query)
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]

    df = pd.DataFrame.from_records(rows, columns=columns)
    df['date'] = pd.to_datetime(df['date'])

    print(f"Downloaded {len(df):,} rows")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Regions: {df['region'].nunique()}")

    os.makedirs('data', exist_ok=True)
    df.to_parquet('data/unified_data_with_forex.parquet', index=False)
    print("Saved to data/unified_data_with_forex.parquet")

    conn.close()
    return df


def pivot_regional_weather(df, feature_set):
    """
    Pivot regional data so each region's weather becomes separate features.

    Args:
        df: Full dataframe with all regions
        feature_set: 'weather', 'weather_vix', 'weather_forex', 'all'

    Returns:
        target_series, covariate_series, feature_names
    """
    print(f"\nPivoting regional weather for feature_set: {feature_set}")

    # Define base feature lists
    weather_features = [
        'temp_max_c', 'temp_min_c', 'temp_mean_c',
        'precipitation_mm', 'rain_mm', 'snowfall_cm',
        'humidity_mean_pct', 'wind_speed_max_kmh'
    ]

    forex_features = [
        'vnd_usd', 'cop_usd', 'idr_usd', 'etb_usd', 'hnl_usd',
        'ugx_usd', 'pen_usd', 'xaf_usd', 'gtq_usd', 'gnf_usd',
        'nio_usd', 'crc_usd', 'tzs_usd', 'kes_usd', 'lak_usd',
        'pkr_usd', 'php_usd', 'egp_usd', 'ars_usd', 'rub_usd',
        'try_usd', 'uah_usd', 'irr_usd', 'byn_usd'
    ]

    # Get unique regions
    regions = sorted(df['region'].unique())
    print(f"Found {len(regions)} regions: {regions[:5]}... (showing first 5)")

    # PIVOT WEATHER: Create region-specific weather features
    pivoted_weather_dfs = []
    for region in regions:
        df_region = df[df['region'] == region][['date'] + weather_features].copy()

        # Rename columns to include region
        region_clean = region.lower().replace(' ', '_').replace(',', '')
        for col in weather_features:
            df_region[f'{col}_{region_clean}'] = df_region[col]
            df_region = df_region.drop(col, axis=1)

        pivoted_weather_dfs.append(df_region)

    # Merge all regional weather into single dataframe
    df_pivoted = pivoted_weather_dfs[0]
    for region_df in pivoted_weather_dfs[1:]:
        df_pivoted = df_pivoted.merge(region_df, on='date', how='outer')

    # Add VIX (global, not regional)
    if feature_set in ['weather_vix', 'all']:
        df_vix = df.groupby('date')['vix'].mean().reset_index()
        df_pivoted = df_pivoted.merge(df_vix, on='date', how='left')

    # Add forex (global averages, not regional)
    if feature_set in ['weather_forex', 'all']:
        df_forex = df.groupby('date')[forex_features].mean().reset_index()
        df_pivoted = df_pivoted.merge(df_forex, on='date', how='left')

    # Add target (global coffee price - average across regions)
    df_target = df.groupby('date')['close'].mean().reset_index()
    df_pivoted = df_pivoted.merge(df_target, on='date', how='left')

    # Sort and forward-fill
    df_pivoted = df_pivoted.sort_values('date').reset_index(drop=True)
    df_pivoted = df_pivoted.fillna(method='ffill').fillna(0)

    # Identify covariate columns
    covariate_cols = [col for col in df_pivoted.columns if col not in ['date', 'close']]

    print(f"Created {len(covariate_cols)} pivoted features")
    print(f"Sample features: {covariate_cols[:5]}...")

    # Create TimeSeries
    target = TimeSeries.from_dataframe(
        df_pivoted,
        time_col='date',
        value_cols='close',
        freq='D',
        fill_missing_dates=True
    )

    covariates = TimeSeries.from_dataframe(
        df_pivoted,
        time_col='date',
        value_cols=covariate_cols,
        freq='D',
        fill_missing_dates=True
    )

    return target, covariates, covariate_cols


def train_and_evaluate(model_name, target, covariates, train_size, horizon, input_chunk_length=60):
    """
    Train model and evaluate at specified horizon using walk-forward validation.
    """
    # Split data
    train_target = target[:train_size]
    val_size = len(target) - train_size

    # Create model
    if model_name == 'NHiTS':
        model = NHiTSModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=horizon,
            num_stacks=3,
            num_blocks=1,
            num_layers=2,
            layer_widths=512,
            n_epochs=100,
            batch_size=32,
            pl_trainer_kwargs={"accelerator": "cpu"},
            random_state=42,
            force_reset=True,
            save_checkpoints=False
        )
    elif model_name == 'NBEATS':
        model = NBEATSModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=horizon,
            generic_architecture=True,
            num_stacks=30,
            num_blocks=1,
            num_layers=4,
            layer_widths=256,
            n_epochs=100,
            batch_size=32,
            pl_trainer_kwargs={"accelerator": "cpu"},
            random_state=42,
            force_reset=True,
            save_checkpoints=False
        )
    elif model_name == 'TCN':
        model = TCNModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=horizon,
            kernel_size=3,
            num_filters=64,
            n_epochs=100,
            batch_size=32,
            pl_trainer_kwargs={"accelerator": "cpu"},
            random_state=42,
            force_reset=True,
            save_checkpoints=False
        )
    elif model_name == 'Transformer':
        model = TransformerModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=horizon,
            d_model=64,
            nhead=4,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=256,
            n_epochs=100,
            batch_size=32,
            pl_trainer_kwargs={"accelerator": "cpu"},
            random_state=42,
            force_reset=True,
            save_checkpoints=False
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Train
    print(f"    Training {model_name} for {horizon}-day horizon...")
    try:
        model.fit(
            series=train_target,
            past_covariates=covariates[:train_size]
        )

        # Walk-forward validation
        print(f"    Running walk-forward validation ({val_size} days)...")
        historical_forecasts = model.historical_forecasts(
            series=target,
            past_covariates=covariates,
            start=train_size,
            forecast_horizon=horizon,
            stride=horizon,
            retrain=False,
            verbose=False
        )

        # Get actual values
        val_target = target[train_size:]

        # Calculate metrics
        mape_score = mape(val_target, historical_forecasts)
        rmse_score = rmse(val_target, historical_forecasts)
        mae_score = mae(val_target, historical_forecasts)
        num_windows = len(historical_forecasts) // horizon

        return {
            'success': True,
            'mape': float(mape_score),
            'rmse': float(rmse_score),
            'mae': float(mae_score),
            'num_windows': num_windows
        }

    except Exception as e:
        print(f"      ERROR: {e}")
        return {
            'success': False,
            'error': str(e),
            'mape': np.nan,
            'rmse': np.nan,
            'mae': np.nan,
            'num_windows': 0
        }


def run_multi_regional_experiments():
    """Run experiments with pivoted regional weather features."""
    print("\n" + "=" * 80)
    print("MULTI-REGIONAL PIVOT EXPERIMENT")
    print("=" * 80)
    print()

    # Download fresh data
    if not os.path.exists('data/unified_data_with_forex.parquet'):
        df = download_fresh_data()
    else:
        print("Loading cached data...")
        df = pd.read_parquet('data/unified_data_with_forex.parquet')
        print(f"Loaded {len(df):,} rows")
        print(f"Regions: {df['region'].nunique()}")

    # Experiment configuration
    models = ['NHiTS', 'NBEATS', 'TCN', 'Transformer']
    feature_sets = ['weather', 'weather_vix', 'weather_forex', 'all']
    horizons = [1, 3, 7, 14]

    results = []
    total_experiments = len(models) * len(feature_sets) * len(horizons)
    experiment_num = 0

    print(f"\nPlanned experiments: {total_experiments}")
    print(f"  Models: {models}")
    print(f"  Feature sets: {feature_sets} (all with PIVOTED regional weather)")
    print(f"  Horizons: {horizons}")
    print()

    # Run experiments
    for feature_set in feature_sets:
        print(f"\n{'='*80}")
        print(f"FEATURE SET: {feature_set} (pivoted across 22 regions)")
        print(f"{'='*80}")

        # Prepare pivoted data once per feature_set
        target, covariates, features = pivot_regional_weather(df, feature_set)

        train_size = int(len(target) * 0.8)
        print(f"Data: {len(target)} days (train: {train_size}, val: {len(target) - train_size})")
        print(f"Features ({len(features)}): {', '.join(features[:5])}{'...' if len(features) > 5 else ''}")

        for model_name in models:
            for horizon in horizons:
                experiment_num += 1
                print(f"  [{experiment_num}/{total_experiments}] {model_name} @ {horizon}-day horizon", end=" ")

                # Train and evaluate
                result = train_and_evaluate(
                    model_name=model_name,
                    target=target,
                    covariates=covariates,
                    train_size=train_size,
                    horizon=horizon
                )

                # Record results
                results.append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'model': model_name,
                    'feature_set': feature_set,
                    'num_features': len(features),
                    'region': 'pivoted_all_22',  # Special marker for pivoted approach
                    'horizon_days': horizon,
                    'train_size': train_size,
                    'val_size': len(target) - train_size,
                    'num_windows': result.get('num_windows', 0),
                    'mape': result['mape'],
                    'rmse': result['rmse'],
                    'mae': result['mae'],
                    'success': result['success']
                })

                if result['success']:
                    print(f"→ MAPE: {result['mape']:.2f}% ({result['num_windows']} windows)")
                else:
                    print(f"→ FAILED")

                # Save results incrementally
                df_results = pd.DataFrame(results)
                df_results.to_csv('experiment_results_pivoted_regional.csv', index=False)

    print("\n" + "=" * 80)
    print("EXPERIMENTS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: experiment_results_pivoted_regional.csv")
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {sum(r['success'] for r in results)}")
    print(f"Failed: {sum(not r['success'] for r in results)}")

    # Summary statistics
    df_results = pd.DataFrame(results)
    df_success = df_results[df_results['success']]

    if len(df_success) > 0:
        print("\n" + "=" * 80)
        print("TOP 10 PERFORMERS BY MAPE")
        print("=" * 80)
        top_10 = df_success.nsmallest(10, 'mape')[
            ['model', 'feature_set', 'num_features', 'horizon_days', 'mape', 'rmse', 'mae']
        ]
        print(top_10.to_string(index=False))

        print("\n" + "=" * 80)
        print("AVERAGE MAPE BY MODEL")
        print("=" * 80)
        avg_by_model = df_success.groupby('model')['mape'].agg(['mean', 'std', 'min', 'max'])
        print(avg_by_model.sort_values('mean'))

        print("\n" + "=" * 80)
        print("AVERAGE MAPE BY FEATURE SET")
        print("=" * 80)
        avg_by_features = df_success.groupby('feature_set')['mape'].agg(['mean', 'std', 'min', 'max'])
        print(avg_by_features.sort_values('mean'))

        print("\n" + "=" * 80)
        print("COMPARISON TO BASELINE (from comprehensive experiments)")
        print("=" * 80)
        print("Best single-region (Bahia): TCN @ 1-day = 1.34% MAPE")
        print("Best aggregated: (pending from running experiments)")
        print(f"Best pivoted (this run): {df_success['mape'].min():.2f}% MAPE")
        print()
        print("If pivoted < single-region: Multi-regional weather IS valuable!")
        print("If pivoted > single-region: Single region weather is sufficient")


if __name__ == '__main__':
    run_multi_regional_experiments()
