"""
Comprehensive DARTS Model Experiments

Tests multiple models, feature combinations, and forecast horizons.
Logs all results to CSV for comparison.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from darts import TimeSeries
from darts.models import NHiTSModel, NBEATSModel, TFTModel, TCNModel, TransformerModel
from darts.metrics import mape, rmse, mae
import warnings
warnings.filterwarnings('ignore')


def download_fresh_data():
    """Download unified_data with all forex columns."""
    print("=" * 80)
    print("DOWNLOADING FRESH DATA WITH FOREX")
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
    print(f"Columns: {len(df.columns)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    os.makedirs('data', exist_ok=True)
    df.to_parquet('data/unified_data_with_forex.parquet', index=False)
    print("Saved to data/unified_data_with_forex.parquet")

    conn.close()
    return df


def prepare_data_for_region(df, region, feature_set):
    """
    Prepare data for a specific region and feature set.

    Args:
        df: Full dataframe
        region: Region name or 'aggregated'
        feature_set: 'weather', 'weather_vix', 'weather_forex', 'all'

    Returns:
        target_series, covariate_series, feature_names
    """
    # Filter or aggregate data
    if region == 'aggregated':
        df_region = df.groupby('date').agg({
            'close': 'mean',
            'volume': 'sum',
            'vix': 'mean',
            # Forex (average across regions)
            'vnd_usd': 'mean', 'cop_usd': 'mean', 'idr_usd': 'mean',
            'etb_usd': 'mean', 'hnl_usd': 'mean', 'ugx_usd': 'mean',
            'pen_usd': 'mean', 'xaf_usd': 'mean', 'gtq_usd': 'mean',
            'gnf_usd': 'mean', 'nio_usd': 'mean', 'crc_usd': 'mean',
            'tzs_usd': 'mean', 'kes_usd': 'mean', 'lak_usd': 'mean',
            'pkr_usd': 'mean', 'php_usd': 'mean', 'egp_usd': 'mean',
            'ars_usd': 'mean', 'rub_usd': 'mean', 'try_usd': 'mean',
            'uah_usd': 'mean', 'irr_usd': 'mean', 'byn_usd': 'mean',
            # Weather (average across regions)
            'temp_max_c': 'mean', 'temp_min_c': 'mean', 'temp_mean_c': 'mean',
            'precipitation_mm': 'mean', 'rain_mm': 'mean', 'snowfall_cm': 'mean',
            'humidity_mean_pct': 'mean', 'wind_speed_max_kmh': 'mean'
        }).reset_index()
    else:
        df_region = df[df['region'] == region].copy()

    df_region = df_region.sort_values('date').reset_index(drop=True)

    # Define feature sets
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

    # Select features based on feature_set
    if feature_set == 'weather':
        features = weather_features
    elif feature_set == 'weather_vix':
        features = weather_features + ['vix']
    elif feature_set == 'weather_forex':
        features = weather_features + forex_features
    elif feature_set == 'all':
        features = weather_features + ['vix'] + forex_features
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    # Fill NaNs
    df_region[features] = df_region[features].fillna(method='ffill').fillna(0)

    # Create TimeSeries
    target = TimeSeries.from_dataframe(
        df_region,
        time_col='date',
        value_cols='close',
        freq='D',
        fill_missing_dates=True
    )

    covariates = TimeSeries.from_dataframe(
        df_region,
        time_col='date',
        value_cols=features,
        freq='D',
        fill_missing_dates=True
    )

    return target, covariates, features


def train_and_evaluate(model_name, target, covariates, train_size, horizon, input_chunk_length=60):
    """
    Train model and evaluate at specified horizon using walk-forward validation.

    Returns:
        dict with metrics and forecasts
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
    elif model_name == 'TFT':
        model = TFTModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=horizon,
            hidden_size=64,
            lstm_layers=1,
            num_attention_heads=4,
            add_relative_index=True,  # ← FIX: Auto-generate future covariates
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

        # Walk-forward validation using historical_forecasts()
        # This evaluates across ENTIRE validation period, not just first 14 days
        print(f"    Running walk-forward validation ({val_size} days)...")
        historical_forecasts = model.historical_forecasts(
            series=target,
            past_covariates=covariates,
            start=train_size,  # Start of validation period
            forecast_horizon=horizon,
            stride=horizon,  # Non-overlapping windows for speed
            retrain=False,  # Use same trained model (faster)
            verbose=False
        )

        # Get actual values for the validation period
        val_target = target[train_size:]

        # Calculate metrics across ALL forecast windows
        mape_score = mape(val_target, historical_forecasts)
        rmse_score = rmse(val_target, historical_forecasts)
        mae_score = mae(val_target, historical_forecasts)

        # Count forecast windows
        num_windows = len(historical_forecasts) // horizon

        return {
            'success': True,
            'mape': float(mape_score),
            'rmse': float(rmse_score),
            'mae': float(mae_score),
            'num_windows': num_windows,
            'forecast': historical_forecasts,
            'actual': val_target
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


def run_comprehensive_experiments():
    """Run comprehensive experiments across models, features, regions, and horizons."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE DARTS EXPERIMENTS")
    print("=" * 80)
    print()

    # Download fresh data
    if not os.path.exists('data/unified_data_with_forex.parquet'):
        df = download_fresh_data()
    else:
        print("Loading cached data...")
        df = pd.read_parquet('data/unified_data_with_forex.parquet')
        print(f"Loaded {len(df):,} rows")

    # Experiment configuration
    models = ['NHiTS', 'NBEATS', 'TCN', 'Transformer', 'TFT']  # ← Added TFT!
    feature_sets = ['weather', 'weather_vix', 'weather_forex', 'all']
    regions = ['Bahia_Brazil', 'aggregated']  # Start with 2 regions
    horizons = [1, 3, 7, 14]  # 1-14 day forecast range

    results = []
    total_experiments = len(models) * len(feature_sets) * len(regions) * len(horizons)
    experiment_num = 0

    print(f"\nPlanned experiments: {total_experiments}")
    print(f"  Models: {models}")
    print(f"  Feature sets: {feature_sets}")
    print(f"  Regions: {regions}")
    print(f"  Horizons: {horizons}")
    print()

    # Run experiments
    for region in regions:
        print(f"\n{'='*80}")
        print(f"REGION: {region}")
        print(f"{'='*80}")

        for feature_set in feature_sets:
            print(f"\n  Feature Set: {feature_set}")

            # Prepare data once per region/feature_set
            target, covariates, features = prepare_data_for_region(df, region, feature_set)

            train_size = int(len(target) * 0.8)
            print(f"  Data: {len(target)} days (train: {train_size}, val: {len(target) - train_size})")
            print(f"  Features ({len(features)}): {', '.join(features[:5])}{'...' if len(features) > 5 else ''}")

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
                        'region': region,
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
                    df_results.to_csv('experiment_results_comprehensive.csv', index=False)

    print("\n" + "=" * 80)
    print("EXPERIMENTS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: experiment_results_comprehensive.csv")
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {sum(r['success'] for r in results)}")
    print(f"Failed: {sum(not r['success'] for r in results)}")

    # Summary statistics
    df_results = pd.DataFrame(results)
    df_success = df_results[df_results['success']]

    if len(df_success) > 0:
        print("\n" + "=" * 80)
        print("TOP PERFORMERS BY MAPE")
        print("=" * 80)
        top_10 = df_success.nsmallest(10, 'mape')[
            ['model', 'feature_set', 'region', 'horizon_days', 'mape', 'rmse', 'mae']
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
        print("AVERAGE MAPE BY HORIZON")
        print("=" * 80)
        avg_by_horizon = df_success.groupby('horizon_days')['mape'].agg(['mean', 'std', 'min', 'max'])
        print(avg_by_horizon.sort_values('horizon_days'))


if __name__ == '__main__':
    run_comprehensive_experiments()
