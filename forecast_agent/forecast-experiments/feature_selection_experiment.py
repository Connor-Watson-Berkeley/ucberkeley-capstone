"""
Feature Selection Experiment

Tests whether intelligent feature selection can improve forecasting by:
1. GDELT Sentiment: Remove problematic features causing NaN
2. Regional Pivot: Reduce 176 features to top performers

Methods:
- Tree-based importance (XGBoost)
- Correlation filtering
- Variance thresholding
- Top-K selection

Goal: Beat naive baseline (1.27% MAPE) with curated features
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# DARTS imports
from darts import TimeSeries
from darts.models import TCNModel, NHiTSModel
from darts.metrics import mape, rmse, mae

# Feature selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler

print("=" * 80)
print("FEATURE SELECTION EXPERIMENT")
print("=" * 80)
print()

# ============================================================================
# EXPERIMENT 1: GDELT Sentiment Feature Selection
# ============================================================================

print("=" * 80)
print("EXPERIMENT 1: GDELT SENTIMENT FEATURE SELECTION")
print("=" * 80)
print()

# Load data
df_sentiment = pd.read_parquet('data/unified_data_with_sentiment.parquet')
df_agg = df_sentiment.groupby('date').agg({
    'close': 'mean',
    # Weather
    'temp_max_c': 'mean',
    'temp_min_c': 'mean',
    'temp_mean_c': 'mean',
    'precipitation_mm': 'mean',
    'rain_mm': 'mean',
    'snowfall_cm': 'mean',
    'humidity_mean_pct': 'mean',
    'wind_speed_max_kmh': 'mean',
    'vix': 'mean',
    # GDELT
    **{col: 'mean' for col in df_sentiment.columns if col.startswith('group_') or col.startswith('theme_')}
}).reset_index().sort_values('date')

# Filter to GDELT coverage dates
gdelt_cols = [c for c in df_agg.columns if c.startswith('group_') or c.startswith('theme_')]
df_agg = df_agg[df_agg[gdelt_cols[0]].notna()].copy()

print(f"Data: {len(df_agg)} days with GDELT")
print(f"GDELT features: {len(gdelt_cols)}")
print()

# Feature selection on GDELT features
weather_features = ['temp_max_c', 'temp_min_c', 'temp_mean_c',
                    'precipitation_mm', 'rain_mm', 'snowfall_cm',
                    'humidity_mean_pct', 'wind_speed_max_kmh', 'vix']

# Create feature matrix and target for selection
X = df_agg[gdelt_cols].values
y = df_agg['close'].shift(-1).fillna(method='ffill').values  # Next day price

# Remove any remaining NaN/inf
mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
X = X[mask]
y = y[mask]

print("Feature selection methods:")
print()

# Method 1: Variance Threshold (remove low-variance features)
print("1. Variance Threshold")
var_threshold = VarianceThreshold(threshold=0.01)
var_threshold.fit(X)
selected_variance = np.array(gdelt_cols)[var_threshold.get_support()]
print(f"   Selected {len(selected_variance)} / {len(gdelt_cols)} features")
print(f"   Removed {len(gdelt_cols) - len(selected_variance)} low-variance features")

# Method 2: Correlation with target
print("\n2. Correlation with Target")
correlations = []
for i, col in enumerate(gdelt_cols):
    corr = np.corrcoef(X[:, i], y)[0, 1]
    correlations.append(abs(corr))

top_k = 10
top_corr_idx = np.argsort(correlations)[-top_k:]
selected_corr = np.array(gdelt_cols)[top_corr_idx]
print(f"   Top {top_k} features by correlation:")
for idx in top_corr_idx[::-1]:
    print(f"     {gdelt_cols[idx]}: {correlations[idx]:.4f}")

# Method 3: Random Forest Feature Importance
print("\n3. Random Forest Importance")
rf = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
rf.fit(X, y)
importances = rf.feature_importances_
top_rf_idx = np.argsort(importances)[-top_k:]
selected_rf = np.array(gdelt_cols)[top_rf_idx]
print(f"   Top {top_k} features by importance:")
for idx in top_rf_idx[::-1]:
    print(f"     {gdelt_cols[idx]}: {importances[idx]:.4f}")

# Combined selection: intersection of top features from all methods
selected_gdelt = list(set(selected_variance) & set(selected_corr) & set(selected_rf))
if len(selected_gdelt) < 5:
    # If intersection too small, use union of correlation + RF
    selected_gdelt = list(set(selected_corr) | set(selected_rf))

print(f"\n✓ Final GDELT selection: {len(selected_gdelt)} features")
print(f"  Features: {', '.join(selected_gdelt)}")
print()

# ============================================================================
# EXPERIMENT 2: Regional Pivot Feature Selection
# ============================================================================

print("=" * 80)
print("EXPERIMENT 2: REGIONAL PIVOT FEATURE SELECTION")
print("=" * 80)
print()

# Load pivot data
df_pivot = pd.read_parquet('data/unified_data.parquet')

# Create pivot features
regions = df_pivot['region'].unique()
print(f"Regions: {len(regions)}")

df_pivot_wide = df_pivot.pivot_table(
    index='date',
    columns='region',
    values=['temp_max_c', 'temp_min_c', 'temp_mean_c', 'precipitation_mm',
            'rain_mm', 'snowfall_cm', 'humidity_mean_pct', 'wind_speed_max_kmh']
).reset_index()

df_pivot_wide.columns = ['_'.join(col).strip('_') if col[0] != 'date' else 'date'
                          for col in df_pivot_wide.columns.values]

# Add aggregated target
df_target = df_pivot.groupby('date')['close'].mean().reset_index()
df_pivot_full = df_pivot_wide.merge(df_target, on='date')

# Get VIX
df_vix = df_pivot.groupby('date')['vix'].mean().reset_index()
df_pivot_full = df_pivot_full.merge(df_vix, on='date')

print(f"Pivot features: {len(df_pivot_full.columns) - 2}")  # Exclude date, close
print()

# Feature selection on pivot features
pivot_features = [c for c in df_pivot_full.columns if c not in ['date', 'close']]
X_pivot = df_pivot_full[pivot_features].values
y_pivot = df_pivot_full['close'].shift(-1).fillna(method='ffill').values

# Remove NaN/inf
mask_pivot = np.isfinite(X_pivot).all(axis=1) & np.isfinite(y_pivot)
X_pivot = X_pivot[mask_pivot]
y_pivot = y_pivot[mask_pivot]

print("Feature selection methods:")
print()

# Method 1: Variance Threshold
print("1. Variance Threshold")
var_threshold_pivot = VarianceThreshold(threshold=0.1)
var_threshold_pivot.fit(X_pivot)
selected_variance_pivot = np.array(pivot_features)[var_threshold_pivot.get_support()]
print(f"   Selected {len(selected_variance_pivot)} / {len(pivot_features)} features")

# Method 2: Correlation filtering (remove highly correlated features)
print("\n2. Correlation Matrix (remove redundant features)")
corr_matrix = np.corrcoef(X_pivot.T)
upper_tri = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
high_corr_pairs = np.where((abs(corr_matrix) > 0.95) & upper_tri)
to_drop = set()
for i, j in zip(*high_corr_pairs):
    to_drop.add(j)  # Drop second feature in pair

selected_corr_pivot = [f for i, f in enumerate(pivot_features) if i not in to_drop]
print(f"   Removed {len(to_drop)} highly correlated features")
print(f"   Kept {len(selected_corr_pivot)} features")

# Method 3: Random Forest Importance (top 30)
print("\n3. Random Forest Importance (Top 30)")
rf_pivot = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
rf_pivot.fit(X_pivot, y_pivot)
importances_pivot = rf_pivot.feature_importances_
top_30_idx = np.argsort(importances_pivot)[-30:]
selected_rf_pivot = np.array(pivot_features)[top_30_idx]
print(f"   Top 30 features:")
for idx in top_30_idx[::-1][:10]:  # Show top 10
    print(f"     {pivot_features[idx]}: {importances_pivot[idx]:.4f}")

# Combined selection
selected_pivot = list(set(selected_variance_pivot) & set(selected_corr_pivot) & set(selected_rf_pivot))
if len(selected_pivot) < 20:
    selected_pivot = list(set(selected_rf_pivot) & set(selected_corr_pivot))

print(f"\n✓ Final Pivot selection: {len(selected_pivot)} features")
print()

# ============================================================================
# EXPERIMENT 3: Train Models with Selected Features
# ============================================================================

print("=" * 80)
print("EXPERIMENT 3: TRAIN MODELS WITH SELECTED FEATURES")
print("=" * 80)
print()

results = []

# Test 1: Weather + Selected GDELT (vs weather-only baseline)
print("Test 1: Weather + Selected GDELT features")
print("-" * 80)

# Prepare data
selected_features_gdelt = weather_features + selected_gdelt
df_test = df_agg[['date', 'close'] + selected_features_gdelt].dropna()

target = TimeSeries.from_dataframe(df_test, time_col='date', value_cols='close')
covariates = TimeSeries.from_dataframe(df_test, time_col='date', value_cols=selected_features_gdelt)

train_size = int(len(target) * 0.8)
print(f"Train: {train_size}, Val: {len(target) - train_size}")
print(f"Features: {len(selected_features_gdelt)}")
print()

# Train TCN
print("  Training TCN...")
model = TCNModel(
    input_chunk_length=60,
    output_chunk_length=7,
    n_epochs=50,
    batch_size=32,
    random_state=42,
    force_reset=True,
    save_checkpoints=False,
    pl_trainer_kwargs={"accelerator": "cpu"}
)

try:
    model.fit(series=target[:train_size], past_covariates=covariates[:train_size])

    forecasts = model.historical_forecasts(
        series=target,
        past_covariates=covariates,
        start=train_size,
        forecast_horizon=1,
        stride=1,
        retrain=False,
        verbose=False
    )

    val_target = target[train_size:]
    mape_score = mape(val_target, forecasts)
    rmse_score = rmse(val_target, forecasts)
    mae_score = mae(val_target, forecasts)

    print(f"  ✓ TCN + Selected GDELT: MAPE={mape_score:.2f}%, RMSE={rmse_score:.2f}, MAE={mae_score:.2f}")

    results.append({
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model': 'TCN',
        'feature_set': 'weather_gdelt_selected',
        'num_features': len(selected_features_gdelt),
        'horizon_days': 1,
        'mape': float(mape_score),
        'rmse': float(rmse_score),
        'mae': float(mae_score),
        'success': True
    })

except Exception as e:
    print(f"  ✗ TCN failed: {e}")
    results.append({
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model': 'TCN',
        'feature_set': 'weather_gdelt_selected',
        'num_features': len(selected_features_gdelt),
        'horizon_days': 1,
        'mape': np.nan,
        'rmse': np.nan,
        'mae': np.nan,
        'success': False,
        'error': str(e)
    })

print()

# Test 2: Selected Pivot Features
print("Test 2: Selected Regional Pivot features")
print("-" * 80)

df_pivot_test = df_pivot_full[['date', 'close'] + selected_pivot].dropna()

target_pivot = TimeSeries.from_dataframe(df_pivot_test, time_col='date', value_cols='close')
covariates_pivot = TimeSeries.from_dataframe(df_pivot_test, time_col='date', value_cols=selected_pivot)

train_size_pivot = int(len(target_pivot) * 0.8)
print(f"Train: {train_size_pivot}, Val: {len(target_pivot) - train_size_pivot}")
print(f"Features: {len(selected_pivot)}")
print()

# Train TCN
print("  Training TCN...")
model_pivot = TCNModel(
    input_chunk_length=60,
    output_chunk_length=7,
    n_epochs=50,
    batch_size=32,
    random_state=42,
    force_reset=True,
    save_checkpoints=False,
    pl_trainer_kwargs={"accelerator": "cpu"}
)

try:
    model_pivot.fit(series=target_pivot[:train_size_pivot], past_covariates=covariates_pivot[:train_size_pivot])

    forecasts_pivot = model_pivot.historical_forecasts(
        series=target_pivot,
        past_covariates=covariates_pivot,
        start=train_size_pivot,
        forecast_horizon=1,
        stride=1,
        retrain=False,
        verbose=False
    )

    val_target_pivot = target_pivot[train_size_pivot:]
    mape_score_pivot = mape(val_target_pivot, forecasts_pivot)
    rmse_score_pivot = rmse(val_target_pivot, forecasts_pivot)
    mae_score_pivot = mae(val_target_pivot, forecasts_pivot)

    print(f"  ✓ TCN + Selected Pivot: MAPE={mape_score_pivot:.2f}%, RMSE={rmse_score_pivot:.2f}, MAE={mae_score_pivot:.2f}")

    results.append({
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model': 'TCN',
        'feature_set': 'pivot_selected',
        'num_features': len(selected_pivot),
        'horizon_days': 1,
        'mape': float(mape_score_pivot),
        'rmse': float(rmse_score_pivot),
        'mae': float(mae_score_pivot),
        'success': True
    })

except Exception as e:
    print(f"  ✗ TCN failed: {e}")
    results.append({
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model': 'TCN',
        'feature_set': 'pivot_selected',
        'num_features': len(selected_pivot),
        'horizon_days': 1,
        'mape': np.nan,
        'rmse': np.nan,
        'mae': np.nan,
        'success': False,
        'error': str(e)
    })

print()

# Save results
df_results = pd.DataFrame(results)
df_results.to_csv('experiment_results_feature_selection.csv', index=False)

print("=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)
print()

print("Baseline comparisons:")
print(f"  Naive baseline:        1.27% MAPE")
print(f"  TCN weather-only:      1.30% MAPE")
print(f"  TCN pivot (176 feat):  1.74% MAPE")
print()

if len(df_results[df_results['success']]) > 0:
    print("Feature selection results:")
    for _, row in df_results[df_results['success']].iterrows():
        improvement = 1.30 - row['mape']  # vs weather baseline
        print(f"  {row['feature_set']}: {row['mape']:.2f}% MAPE ({row['num_features']} features)")
        if row['mape'] < 1.27:
            print(f"    🎉 BEATS NAIVE BASELINE by {1.27 - row['mape']:.2f}%")
        elif row['mape'] < 1.30:
            print(f"    ✓ Improves on weather baseline by {improvement:.2f}%")
        else:
            print(f"    ✗ Worse than baseline by {-improvement:.2f}%")
    print()

print(f"Results saved to: experiment_results_feature_selection.csv")
