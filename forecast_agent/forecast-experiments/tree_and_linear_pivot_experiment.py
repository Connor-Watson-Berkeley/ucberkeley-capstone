"""
Tree-Based and Regularized Linear Models on Multi-Regional Pivot Data

Tests XGBoost, LightGBM, LASSO, Ridge, and ElasticNet on the 176-feature
pivoted regional weather data.

These models handle high dimensionality better than deep learning:
- Tree models: Can ignore irrelevant features
- Regularized linear: Shrink/zero-out unimportant coefficients
- Both: Built-in feature selection
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML imports
from xgboost import XGBRegressor
try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("Warning: LightGBM not installed, skipping")
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error


def load_pivoted_data():
    """Load and pivot regional weather data."""
    print("=" * 80)
    print("LOADING AND PIVOTING REGIONAL DATA")
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

    cursor.execute(query)
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]

    df = pd.DataFrame.from_records(rows, columns=columns)
    df['date'] = pd.to_datetime(df['date'])
    conn.close()

    # Pivot weather by region
    weather_features = [
        'temp_max_c', 'temp_min_c', 'temp_mean_c',
        'precipitation_mm', 'rain_mm', 'snowfall_cm',
        'humidity_mean_pct', 'wind_speed_max_kmh'
    ]

    regions = sorted(df['region'].unique())
    print(f"Pivoting {len(regions)} regions...")

    pivoted_dfs = []
    for region in regions:
        df_region = df[df['region'] == region][['date'] + weather_features].copy()
        region_clean = region.lower().replace(' ', '_').replace(',', '')

        for col in weather_features:
            df_region[f'{col}_{region_clean}'] = df_region[col]
            df_region = df_region.drop(col, axis=1)

        pivoted_dfs.append(df_region)

    df_pivoted = pivoted_dfs[0]
    for region_df in pivoted_dfs[1:]:
        df_pivoted = df_pivoted.merge(region_df, on='date', how='outer')

    # Add VIX
    df_vix = df.groupby('date')['vix'].mean().reset_index()
    df_pivoted = df_pivoted.merge(df_vix, on='date', how='left')

    # Add forex
    forex_features = [
        'vnd_usd', 'cop_usd', 'idr_usd', 'etb_usd', 'hnl_usd',
        'ugx_usd', 'pen_usd', 'xaf_usd', 'gtq_usd', 'gnf_usd',
        'nio_usd', 'crc_usd', 'tzs_usd', 'kes_usd', 'lak_usd',
        'pkr_usd', 'php_usd', 'egp_usd', 'ars_usd', 'rub_usd',
        'try_usd', 'uah_usd', 'irr_usd', 'byn_usd'
    ]
    df_forex = df.groupby('date')[forex_features].mean().reset_index()
    df_pivoted = df_pivoted.merge(df_forex, on='date', how='left')

    # Add target
    df_target = df.groupby('date')['close'].mean().reset_index()
    df_pivoted = df_pivoted.merge(df_target, on='date', how='left')

    # Forward-fill
    df_pivoted = df_pivoted.sort_values('date').fillna(method='ffill').fillna(0)

    print(f"Pivoted data shape: {df_pivoted.shape}")
    print(f"Total features: {df_pivoted.shape[1] - 2} (excluding date and close)")

    return df_pivoted


def create_lagged_features(df, feature_cols, target_col, lag_days=60):
    """
    Create lagged features for time series forecasting.

    For each feature, create lag_days previous values.
    This allows traditional ML models to learn temporal patterns.
    """
    print(f"Creating {lag_days}-day lagged features...")

    df_lagged = df.copy()

    # Create lags for each feature
    for col in feature_cols:
        for lag in range(1, lag_days + 1):
            df_lagged[f'{col}_lag{lag}'] = df[col].shift(lag)

    # Create lags for target (autoregressive)
    for lag in range(1, lag_days + 1):
        df_lagged[f'{target_col}_lag{lag}'] = df[target_col].shift(lag)

    # Drop rows with NaN (first lag_days rows)
    df_lagged = df_lagged.dropna()

    lagged_feature_cols = [c for c in df_lagged.columns
                          if c not in ['date', target_col] and '_lag' in c]

    print(f"Created {len(lagged_feature_cols)} lagged features")

    return df_lagged, lagged_feature_cols


def walk_forward_validation(model, X, y, train_size, horizon, stride):
    """
    Perform walk-forward validation for time series.

    Train once on train_size, then predict horizon-step-ahead
    for each window in validation set.
    """
    predictions = []
    actuals = []

    # Number of forecast windows
    val_size = len(X) - train_size
    num_windows = (val_size - horizon) // stride + 1

    for i in range(num_windows):
        val_start = train_size + i * stride
        val_end = val_start + horizon

        if val_end > len(X):
            break

        # Predict
        X_forecast = X[val_start:val_end]
        y_forecast = y[val_start:val_end]

        preds = model.predict(X_forecast)

        predictions.extend(preds)
        actuals.extend(y_forecast)

    return np.array(actuals), np.array(predictions)


def train_and_evaluate(model_name, model, X_train, y_train, X_val, y_val,
                      feature_names, horizon=1):
    """Train model and evaluate with walk-forward validation."""
    print(f"\n  Training {model_name}...")

    try:
        # Train
        model.fit(X_train, y_train)

        # Walk-forward validation
        print(f"  Running walk-forward validation (horizon={horizon})...")

        # Simple approach: predict horizon steps ahead from each validation point
        predictions = []
        actuals = []

        # For each point in validation, predict next 'horizon' steps
        for i in range(0, len(X_val) - horizon, horizon):
            X_window = X_val[i:i+1]  # Use single point
            y_window = y_val[i+horizon-1:i+horizon]  # Target is 'horizon' steps ahead

            if len(y_window) == 0:
                break

            pred = model.predict(X_window)[0]
            predictions.append(pred)
            actuals.append(y_window[0])

        # Calculate metrics
        actuals = np.array(actuals)
        predictions = np.array(predictions)

        mape = mean_absolute_percentage_error(actuals, predictions) * 100
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)

        print(f"  ✓ {model_name}: MAPE={mape:.2f}%, RMSE={rmse:.2f}, MAE={mae:.2f}")

        # Get feature importances if available
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(feature_names, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            feature_importance = dict(zip(feature_names, np.abs(model.coef_)))

        return {
            'success': True,
            'mape': float(mape),
            'rmse': float(rmse),
            'mae': float(mae),
            'num_windows': len(predictions),
            'feature_importance': feature_importance
        }

    except Exception as e:
        print(f"  ✗ {model_name} failed: {e}")
        return {
            'success': False,
            'mape': np.nan,
            'rmse': np.nan,
            'mae': np.nan,
            'num_windows': 0,
            'error': str(e)
        }


def run_tree_linear_experiments():
    """Run experiments with tree-based and regularized linear models."""
    print("\n" + "=" * 80)
    print("TREE-BASED & LINEAR MODELS ON MULTI-REGIONAL PIVOT DATA")
    print("=" * 80)
    print()

    # Load data
    df = load_pivoted_data()

    # Prepare features
    feature_cols = [c for c in df.columns if c not in ['date', 'close']]

    # For traditional ML, create lagged features (use past 60 days)
    df_lagged, lagged_cols = create_lagged_features(df, feature_cols, 'close', lag_days=60)

    X = df_lagged[lagged_cols].values
    y = df_lagged['close'].values

    # Train/val split
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    print(f"\nData ready:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val: {len(X_val)} samples")
    print(f"  Features: {len(lagged_cols)} (with lags)")
    print()

    # Define models
    models = {
        'XGBoost': XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ),
        'LASSO': Lasso(
            alpha=0.1,
            max_iter=10000,
            random_state=42
        ),
        'Ridge': Ridge(
            alpha=1.0,
            max_iter=10000,
            random_state=42
        ),
        'ElasticNet': ElasticNet(
            alpha=0.1,
            l1_ratio=0.5,
            max_iter=10000,
            random_state=42
        )
    }

    # Add LightGBM if available
    if HAS_LIGHTGBM:
        models['LightGBM'] = LGBMRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )

    horizons = [1, 3, 7, 14]
    results = []

    # Run experiments
    for horizon in horizons:
        print(f"\n{'='*80}")
        print(f"HORIZON: {horizon}-day")
        print(f"{'='*80}")

        for model_name, model in models.items():
            result = train_and_evaluate(
                model_name, model,
                X_train, y_train,
                X_val, y_val,
                lagged_cols,
                horizon=horizon
            )

            # Record results
            results.append({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model': model_name,
                'feature_set': 'all_pivot_lagged',
                'num_features': len(lagged_cols),
                'region': 'pivoted_all_22',
                'horizon_days': horizon,
                'train_size': len(X_train),
                'val_size': len(X_val),
                'num_windows': result.get('num_windows', 0),
                'mape': result['mape'],
                'rmse': result['rmse'],
                'mae': result['mae'],
                'success': result['success']
            })

            # Save feature importances if available
            if result.get('feature_importance'):
                import json
                filename = f"feature_importance_{model_name}_h{horizon}.json"
                with open(filename, 'w') as f:
                    # Save top 50 features
                    top_features = sorted(
                        result['feature_importance'].items(),
                        key=lambda x: x[1], reverse=True
                    )[:50]
                    # Convert numpy types to Python types for JSON serialization
                    top_features_serializable = {feat: float(imp) for feat, imp in top_features}
                    json.dump(top_features_serializable, f, indent=2)
                print(f"    Saved top 50 features to {filename}")

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv('experiment_results_tree_linear_pivot.csv', index=False)

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    df_success = df_results[df_results['success']]

    if len(df_success) > 0:
        print("\nTop 10 by MAPE:")
        top_10 = df_success.nsmallest(10, 'mape')[
            ['model', 'horizon_days', 'mape', 'rmse', 'mae']
        ]
        print(top_10.to_string(index=False))

        print("\n\nAverage MAPE by model:")
        avg_by_model = df_success.groupby('model')['mape'].agg(['mean', 'std', 'min', 'max'])
        print(avg_by_model.sort_values('mean'))

        print("\n\nComparison to deep learning (from pivot experiment):")
        print("  Deep Learning (NHiTS @ 1-day): 9.03% MAPE")
        print(f"  Best Tree/Linear: {df_success['mape'].min():.2f}% MAPE")

        if df_success['mape'].min() < 9.03:
            print("\n  ✓ Tree/Linear models BEAT deep learning on high-dim data!")
        else:
            print("\n  Deep learning still competitive")

    print(f"\nResults saved to: experiment_results_tree_linear_pivot.csv")
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {sum(r['success'] for r in results)}")


if __name__ == '__main__':
    run_tree_linear_experiments()
