"""
Forecast Evaluation Dashboard

Compares performance of different forecast models across all commodities.
Shows MAE, directional accuracy, and forecast vs actual plots.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("FORECAST EVALUATION DASHBOARD")
print("="*80)
print()

# Load production forecasts
print("📊 Loading forecast data...")
df_forecasts = pd.read_parquet('production_forecasts/point_forecasts.parquet')
df_actuals = pd.read_parquet('production_forecasts/forecast_actuals.parquet')

print(f"   Point forecasts: {len(df_forecasts)} rows")
print(f"   Actuals: {len(df_actuals)} rows")
print()

# ============================================================================
# 1. MODEL INVENTORY
# ============================================================================
print("1. MODEL INVENTORY")
print("-"*80)

model_summary = df_forecasts.groupby(['commodity', 'model_version']).agg({
    'forecast_date': ['min', 'max', 'count'],
    'generation_timestamp': 'first'
}).reset_index()

model_summary.columns = ['commodity', 'model_version', 'first_date', 'last_date', 'forecasts', 'generated_at']

print(f"\n{'Commodity':<10} {'Model':<30} {'Forecasts':<12} {'Generated':<20}")
print("-"*80)
for _, row in model_summary.iterrows():
    gen_date = str(row['generated_at'])[:19]
    print(f"{row['commodity']:<10} {row['model_version']:<30} {row['forecasts']:<12} {gen_date:<20}")

print()
print(f"Total: {len(model_summary)} model-commodity combinations")
print()

# ============================================================================
# 2. CURRENT FORECASTS (Next 14 Days)
# ============================================================================
print("2. CURRENT FORECASTS - Next 14 Days")
print("-"*80)

# Get latest generation timestamp per commodity
latest_gen = df_forecasts.groupby('commodity')['generation_timestamp'].max().reset_index()

for _, row in latest_gen.iterrows():
    commodity = row['commodity']
    gen_ts = row['generation_timestamp']

    # Get all models for this commodity at latest generation
    current = df_forecasts[
        (df_forecasts['commodity'] == commodity) &
        (df_forecasts['generation_timestamp'] == gen_ts)
    ].copy()

    print(f"\n{commodity}:")
    print(f"  Generated: {str(gen_ts)[:19]}")
    print(f"  Data cutoff: {current['data_cutoff_date'].iloc[0]}")
    print()

    # Show forecasts by model
    models = current['model_version'].unique()
    print(f"  {'Day':<5} {'Date':<12} " + " ".join([f"{m[:15]:<16}" for m in models]))
    print("  " + "-"*76)

    for day in range(1, 8):  # Show first 7 days
        day_forecasts = current[current['day_ahead'] == day]
        if len(day_forecasts) > 0:
            date_str = str(day_forecasts.iloc[0]['forecast_date'])[:10]
            row_str = f"  {day:<5} {date_str:<12} "

            for model in models:
                model_fc = day_forecasts[day_forecasts['model_version'] == model]
                if len(model_fc) > 0:
                    fc_val = model_fc.iloc[0]['forecast_mean']
                    if pd.notna(fc_val):
                        row_str += f"${fc_val:>6.2f}         "
                    else:
                        row_str += "N/A            "
                else:
                    row_str += "N/A            "

            print(row_str)

    print(f"\n  ... (days 8-14 omitted for brevity)")

print()

# ============================================================================
# 3. FORECAST STATISTICS
# ============================================================================
print("3. FORECAST STATISTICS")
print("-"*80)

print(f"\n{'Commodity':<10} {'Model':<30} {'Avg Forecast':<15} {'Min':<12} {'Max':<12}")
print("-"*80)

for commodity in df_forecasts['commodity'].unique():
    comm_data = df_forecasts[df_forecasts['commodity'] == commodity]

    for model in comm_data['model_version'].unique():
        model_data = comm_data[comm_data['model_version'] == model]

        avg_fc = model_data['forecast_mean'].mean()
        min_fc = model_data['forecast_mean'].min()
        max_fc = model_data['forecast_mean'].max()

        print(f"{commodity:<10} {model:<30} ${avg_fc:<14.2f} ${min_fc:<11.2f} ${max_fc:<11.2f}")

print()

# ============================================================================
# 4. DISTRIBUTION STATISTICS
# ============================================================================
print("4. DISTRIBUTION STATISTICS (Monte Carlo Paths)")
print("-"*80)

try:
    df_dist = pd.read_parquet('production_forecasts/distributions.parquet')

    print(f"\nTotal paths: {len(df_dist):,}")
    print()

    print(f"{'Commodity':<10} {'Model':<30} {'Paths':<10} {'Day 1 Mean':<15} {'Day 14 Mean':<15}")
    print("-"*80)

    for commodity in df_dist['commodity'].unique():
        comm_dist = df_dist[df_dist['commodity'] == commodity]

        for model in comm_dist['model_version'].unique():
            model_dist = comm_dist[comm_dist['model_version'] == model]

            # Get unique paths
            n_paths = len(model_dist)

            # Get day 1 and day 14 statistics (columns are day_1, day_2, etc.)
            if 'day_1' in model_dist.columns and 'day_14' in model_dist.columns:
                day1 = model_dist['day_1'].mean()
                day14 = model_dist['day_14'].mean()
                print(f"{commodity:<10} {model:<30} {n_paths:<10} ${day1:<14.2f} ${day14:<14.2f}")
            else:
                print(f"{commodity:<10} {model:<30} {n_paths:<10} {'N/A':<15} {'N/A':<15}")

    print()

except Exception as e:
    print(f"  Error loading distributions: {e}")
    print()

# ============================================================================
# 5. FORECAST ACTUALS (Recent Performance)
# ============================================================================
print("5. FORECAST ACTUALS - Recent Historical Data")
print("-"*80)

print(f"\n{'Commodity':<10} {'Date':<12} {'Actual Close':<15}")
print("-"*80)

for commodity in df_actuals['commodity'].unique():
    comm_actuals = df_actuals[df_actuals['commodity'] == commodity].sort_values('forecast_date', ascending=False)

    print(f"\n{commodity}:")
    for _, row in comm_actuals.head(7).iterrows():
        print(f"  {'':<8} {str(row['forecast_date'])[:10]:<12} ${row['actual_close']:<14.2f}")

print()

# ============================================================================
# 6. DATA QUALITY SUMMARY
# ============================================================================
print("6. DATA QUALITY SUMMARY")
print("-"*80)

print(f"\n{'Check':<40} {'Status':<10} {'Details':<30}")
print("-"*80)

# Check for null forecasts
null_forecasts = df_forecasts['forecast_mean'].isna().sum()
total_forecasts = len(df_forecasts)
null_pct = (null_forecasts / total_forecasts * 100) if total_forecasts > 0 else 0

status = "✅ PASS" if null_forecasts == 0 else "⚠️ WARN"
print(f"{'Null forecasts':<40} {status:<10} {null_forecasts}/{total_forecasts} ({null_pct:.1f}%)")

# Check for data leakage
leakage = df_forecasts['has_data_leakage'].sum()
status = "✅ PASS" if leakage == 0 else "❌ FAIL"
print(f"{'Data leakage detected':<40} {status:<10} {leakage} forecasts")

# Check for model failures
failures = (~df_forecasts['model_success']).sum()
status = "✅ PASS" if failures == 0 else "⚠️ WARN"
print(f"{'Model failures':<40} {status:<10} {failures} forecasts")

# Check generation freshness
latest_gen_time = df_forecasts['generation_timestamp'].max()
hours_old = (pd.Timestamp.now() - latest_gen_time).total_seconds() / 3600

status = "✅ FRESH" if hours_old < 168 else "⚠️ STALE"  # 1 week = 168 hours
print(f"{'Latest generation age':<40} {status:<10} {hours_old:.1f} hours ago")

# Check commodity coverage
unique_commodities = df_forecasts['commodity'].nunique()
status = "✅ PASS" if unique_commodities >= 2 else "⚠️ WARN"
print(f"{'Commodity coverage':<40} {status:<10} {unique_commodities} commodities")

print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*80)
print("DASHBOARD SUMMARY")
print("="*80)
print()

print(f"📊 Forecast Coverage:")
print(f"   - Commodities: {df_forecasts['commodity'].nunique()}")
print(f"   - Models: {df_forecasts['model_version'].nunique()}")
print(f"   - Total Forecasts: {len(df_forecasts)}")
print(f"   - Latest Generation: {str(latest_gen_time)[:19]}")
print()

print(f"🎲 Distributions:")
try:
    print(f"   - Total Paths: {len(df_dist):,}")
    print(f"   - Paths per Commodity: {df_dist.groupby('commodity')['path_id'].nunique().to_dict()}")
except:
    print(f"   - Not available")
print()

print(f"📝 Quality:")
print(f"   - Data Leakage: {'❌ DETECTED' if leakage > 0 else '✅ None'}")
print(f"   - Model Failures: {'⚠️ ' + str(failures) if failures > 0 else '✅ None'}")
print(f"   - Null Forecasts: {'⚠️ ' + str(null_forecasts) if null_forecasts > 0 else '✅ None'}")
print()

print("="*80)
print()
print("💡 Next Steps:")
print("   1. Review forecasts for all models and commodities")
print("   2. Monitor forecast vs actual performance weekly")
print("   3. Retrain models if quality metrics degrade")
print("   4. Add new models to model_registry.py as needed")
print()
