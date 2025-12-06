# Multi-Regional Pivot Experiment

**Date**: 2025-11-23
**Status**: EXPERIMENTAL - Testing for potential feature engineering integration
**Experiment Script**: `multi_regional_pivot_experiment.py`

---

## Executive Summary

Testing whether **pivoting all 22 coffee-growing regions' weather into separate features** improves forecast accuracy compared to:
1. Single-region approach (e.g., Bahia_Brazil only)
2. Aggregated approach (average weather across all regions)

**Key Question**: Does multi-regional weather signal provide better predictive power?

---

## Motivation

### Current Approaches

**Single-Region (Bahia_Brazil)**:
```python
Features (8):
- temp_max_c
- temp_min_c
- precipitation_mm
- ... (Bahia's weather only)
```
**Result**: 1.34% MAPE (TCN @ 1-day)

**Aggregated (Global Average)**:
```python
Features (8):
- temp_max_c (mean across 22 regions)
- temp_min_c (mean across 22 regions)
- precipitation_mm (mean across 22 regions)
- ... (global averages)
```
**Result**: Pending from comprehensive experiments

### Proposed: Multi-Regional Pivot

**Pivot Approach**:
```python
Features (176):
- temp_max_c_bahia_brazil
- temp_max_c_vietnam
- temp_max_c_colombia
- temp_max_c_ethiopia
- ... (22 regions × 8 weather features)
```

**Hypothesis**: Model can learn which regions' weather is most predictive of global coffee prices.

---

## Coffee Production Context

### Global Coffee Production by Region

| Region | % of Global Production | Key Weather Sensitivity |
|--------|------------------------|------------------------|
| Brazil | 40% | Frost risk (temp_min_c) |
| Vietnam | 20% | Monsoon patterns (precipitation_mm) |
| Colombia | 10% | Stable temps (temp_mean_c) |
| Ethiopia | 5% | Drought risk (precipitation_mm) |
| Honduras | 4% | Hurricane damage (wind, rain) |
| Other 17 | 21% | Various |

**Why This Matters**:
- Brazil frost → massive price impact (40% supply shock)
- Vietnam drought → moderate impact (20% supply)
- Ethiopia drought → minor impact (5% supply)

**Expected Model Behavior**: Should weight Brazil weather features much higher than others.

---

## Technical Implementation

### Data Pivot Process

**Input**: unified_data table (grain: date × region)
```
date       | region         | close | temp_max_c | precipitation_mm | ...
2024-01-01 | Bahia_Brazil   | 180.5 | 32.1       | 5.2              | ...
2024-01-01 | Vietnam        | 180.5 | 28.4       | 12.8             | ...
2024-01-01 | Colombia       | 180.5 | 25.6       | 8.1              | ...
```

**Pivot Operation**:
```python
# For each region, create region-specific feature columns
for region in regions:
    df_region = df[df['region'] == region][['date'] + weather_features]

    # Rename: temp_max_c → temp_max_c_bahia_brazil
    for col in weather_features:
        region_clean = region.lower().replace(' ', '_')
        df_region[f'{col}_{region_clean}'] = df_region[col]

# Merge all regional dataframes on 'date'
df_pivoted = merge_all_on_date(regional_dfs)
```

**Output**: Single dataframe (grain: date only)
```
date       | close | temp_max_c_bahia_brazil | temp_max_c_vietnam | ...
2024-01-01 | 180.5 | 32.1                    | 28.4               | ...
2024-01-02 | 181.2 | 31.8                    | 27.9               | ...
```

### Feature Counts by Set

| Feature Set | # Features | Components |
|-------------|-----------|------------|
| weather | 176 | 22 regions × 8 weather vars |
| weather_vix | 177 | 176 weather + 1 VIX |
| weather_forex | 200 | 176 weather + 24 forex |
| all | 201 | 176 weather + 1 VIX + 24 forex |

### Key Implementation Details

1. **Forward-Fill Missing Data**: Regional weather may have gaps
   ```python
   df_pivoted = df_pivoted.fillna(method='ffill').fillna(0)
   ```

2. **Global Price Target**: Coffee futures price (averaged across regions)
   ```python
   df_target = df.groupby('date')['close'].mean()
   ```

3. **VIX & Forex**: Not pivoted by region (already global metrics)
   ```python
   df_vix = df.groupby('date')['vix'].mean()
   df_forex = df.groupby('date')[forex_features].mean()
   ```

---

## Experiment Configuration

### Models Tested
- NHiTS
- NBEATS
- TCN
- Transformer

### Forecast Horizons
- 1-day
- 3-day
- 7-day
- 14-day

### Validation Method
Walk-forward validation using `historical_forecasts()`:
- Train: 80% of data (3024 days)
- Validation: 20% of data (756 days)
- Stride: horizon (non-overlapping windows)

### Total Experiments
64 experiments (4 models × 4 feature sets × 4 horizons × 1 region config)

---

## Expected Outcomes

### Scenario A: Multi-Regional Wins

**If MAPE < 1.34% (current best single-region)**:

**Interpretation**: Different regions' weather provides complementary signals
- Brazil weather predicts frost risk
- Vietnam weather predicts drought risk
- Model learns to weight regions by production share

**Next Steps**:
1. Feature importance analysis: Which regions matter most?
2. Formalize pivot in feature engineering pipeline
3. Deploy to production forecasting

**Production Integration**:
```python
# In forecast_agent/ground_truth/features/
def create_multi_regional_weather_features(df):
    """Pivot weather by region for global price prediction."""
    pivoted = pivot_weather_by_region(df)
    return pivoted
```

### Scenario B: Single-Region Wins

**If MAPE > 1.34% (worse than single-region)**:

**Interpretation**: Too many features cause overfitting or noise
- Model can't effectively learn regional importance
- Bahia alone captures sufficient signal (as Brazil = 40% of supply)
- Aggregated approach may still beat pivoted

**Next Steps**:
1. Stick with single-region (Bahia_Brazil) approach
2. Consider dimensionality reduction (PCA on pivoted weather)
3. Test selective regional features (Brazil + Vietnam only)

### Scenario C: Aggregated Wins Both

**If Aggregated MAPE < Pivoted < Single-Region**:

**Interpretation**: Global average weather smooths noise better than individual regions
- Random regional fluctuations cancel out
- Global trend is what matters for global price

**Next Steps**:
1. Use aggregated weather in production
2. Monitor for regime changes (e.g., Brazil frost event)
3. Consider hybrid: aggregated + Brazil frost indicator

---

## Feature Importance Analysis (If Successful)

If multi-regional pivot wins, we'll analyze which regions matter:

```python
# Pseudo-code for post-experiment analysis
def analyze_regional_importance(model, pivoted_features):
    """Determine which regions' weather is most predictive."""

    # Get feature importances (model-dependent)
    importances = model.feature_importances_  # For tree-based
    # OR: Use SHAP values for deep learning models

    # Group by region
    regional_scores = {}
    for region in regions:
        region_features = [f for f in pivoted_features if region in f]
        regional_scores[region] = sum(importances[region_features])

    # Rank regions
    return sorted(regional_scores.items(), key=lambda x: x[1], reverse=True)
```

**Expected Top 5**:
1. Bahia_Brazil (40% production)
2. Vietnam (20% production)
3. Colombia (10% production)
4. Central_Highlands_Vietnam (alternative Vietnam region)
5. Minas_Gerais_Brazil (alternative Brazil region)

---

## Integration into Feature Engineering Pipeline

### If Experiment Succeeds

**Location**: `forecast_agent/ground_truth/features/data_preparation.py`

**New Function**:
```python
def pivot_regional_weather(df, feature_set='weather'):
    """
    Pivot weather features by region for multi-regional forecasting.

    Args:
        df: DataFrame with columns [date, region, close, weather features]
        feature_set: 'weather', 'weather_vix', 'weather_forex', 'all'

    Returns:
        DataFrame with pivoted regional weather features (grain: date)
    """
    weather_features = [
        'temp_max_c', 'temp_min_c', 'temp_mean_c',
        'precipitation_mm', 'rain_mm', 'snowfall_cm',
        'humidity_mean_pct', 'wind_speed_max_kmh'
    ]

    regions = sorted(df['region'].unique())

    # Pivot weather by region
    pivoted_dfs = []
    for region in regions:
        df_region = df[df['region'] == region][['date'] + weather_features].copy()
        region_clean = region.lower().replace(' ', '_').replace(',', '')

        for col in weather_features:
            df_region[f'{col}_{region_clean}'] = df_region[col]
            df_region = df_region.drop(col, axis=1)

        pivoted_dfs.append(df_region)

    # Merge all
    df_pivoted = pivoted_dfs[0]
    for region_df in pivoted_dfs[1:]:
        df_pivoted = df_pivoted.merge(region_df, on='date', how='outer')

    # Add non-regional features
    if feature_set in ['weather_vix', 'all']:
        df_vix = df.groupby('date')['vix'].mean().reset_index()
        df_pivoted = df_pivoted.merge(df_vix, on='date', how='left')

    if feature_set in ['weather_forex', 'all']:
        forex_features = [...]  # 24 forex features
        df_forex = df.groupby('date')[forex_features].mean().reset_index()
        df_pivoted = df_pivoted.merge(df_forex, on='date', how='left')

    # Forward-fill and sort
    df_pivoted = df_pivoted.sort_values('date')
    df_pivoted = df_pivoted.fillna(method='ffill').fillna(0)

    return df_pivoted
```

**Usage in Training**:
```python
# In model training scripts
from forecast_agent.ground_truth.features.data_preparation import pivot_regional_weather

# Prepare data
df = load_unified_data()
df_pivoted = pivot_regional_weather(df, feature_set='weather')

# Train model
model.fit(df_pivoted)
```

### Configuration Flag

Add to model config:
```python
# forecast_agent/ground_truth/models/config.py
FEATURE_ENGINEERING = {
    'use_multi_regional_pivot': True,  # ← New flag
    'regional_feature_set': 'weather',  # or 'all'
}
```

---

## Comparison to Alternatives

### Alternative 1: Feature Selection on Pivoted Data

Instead of using all 176 weather features:
1. Run pivot experiment
2. Identify top-N most important regional features
3. Use only those in production

**Pros**: Reduces dimensionality, faster training
**Cons**: Loses long-tail regional signals

### Alternative 2: Hierarchical Modeling

Train separate models per region, ensemble predictions:
```python
predictions = []
for region in regions:
    model = train_on_region(region)
    pred = model.predict()
    predictions.append(pred * production_weights[region])

final_pred = sum(predictions)
```

**Pros**: Each model specializes in regional patterns
**Cons**: More complex infrastructure, harder to deploy

### Alternative 3: Attention Mechanism

Use Transformer/TFT to learn regional importance automatically:
```python
# TFT has built-in variable selection
model = TFTModel(
    static_covariates=['region'],  # Treat region as categorical
    ...
)
```

**Pros**: Model learns importance dynamically
**Cons**: Requires categorical encoding of regions

**Our Pivot Approach**: Simpler, works with all DARTS models, interpretable

---

## Success Metrics

### Primary Metric: MAPE Improvement

**Success Threshold**: Pivoted MAPE < 1.34% (beat single-region baseline)

**Tiers**:
- **Excellent**: < 1.0% MAPE (>25% improvement)
- **Good**: 1.0-1.2% MAPE (10-25% improvement)
- **Marginal**: 1.2-1.34% MAPE (0-10% improvement)
- **Failure**: > 1.34% MAPE (worse than baseline)

### Secondary Metrics

1. **Consistency Across Horizons**: Does multi-regional help at all horizons (1, 3, 7, 14-day)?
2. **Model Universality**: Does it improve all models equally, or just specific architectures?
3. **Feature Set Impact**: Does it help weather-only, or only with forex/VIX too?

---

## Known Risks & Limitations

### 1. Overfitting Risk

**Risk**: 176 features on 3024 training days = high feature/sample ratio
- Weather features: 176
- Training samples: 3024
- Ratio: 1:17 (low for deep learning)

**Mitigation**: Use regularization, dropout, early stopping

### 2. Data Quality Variance

**Risk**: Some regions may have lower-quality weather data
- Missing data handled by forward-fill
- But underlying sensor quality may vary

**Mitigation**: Monitor feature importance - low-quality regions should have low importance

### 3. Computational Cost

**Risk**: 176 features → slower training than 8 features
- Training time may increase 3-5x
- Memory usage increases proportionally

**Mitigation**: Acceptable for research; optimize if deploying to production

### 4. Production Inference

**Risk**: Need real-time weather for 22 regions
- Single-region: Query 1 region's weather
- Multi-regional: Query 22 regions' weather

**Mitigation**: Weather API costs scale linearly; batch queries for efficiency

---

## Timeline

**Experiment Start**: 2025-11-23 22:30 UTC
**Expected Duration**: 4-6 hours (64 experiments, ~4-5 min each)
**Results Available**: 2025-11-24 03:00-05:00 UTC

**If Successful**:
- Day 1: Document findings
- Day 2: Integrate into feature engineering pipeline
- Week 1: Test in production shadow mode
- Week 2: Deploy to production if validates

**If Unsuccessful**:
- Document why it failed
- Test alternative approaches (selective regions, PCA, etc.)

---

## Related Experiments

1. **Comprehensive DARTS Experiments** (`comprehensive_darts_experiments.py`)
   - Tests single-region (Bahia_Brazil) and aggregated approaches
   - Baseline for comparison

2. **Weather Predictive Power Finding** (`WEATHER_PREDICTIVE_POWER_FINDING.md`)
   - Showed weather is extremely predictive (0.1% MAPE with perfect foresight)
   - Validates importance of weather features

3. **TFT Debug Experiment** (`test_tft_debug.py`)
   - Confirmed TFT works with `add_relative_index=True`
   - TFT not included in pivot experiment (focus on proven models)

---

## Appendix: Regional Coffee Production

**22 Regions in unified_data**:

| Region | Country | Approx. % of Global Production |
|--------|---------|-------------------------------|
| Bahia_Brazil | Brazil | 15% |
| Minas_Gerais_Brazil | Brazil | 25% |
| Central_Highlands_Vietnam | Vietnam | 15% |
| Antioquia_Colombia | Colombia | 8% |
| Sidamo_Ethiopia | Ethiopia | 4% |
| Copan_Honduras | Honduras | 3% |
| ... | ... | ... |

*(Full list available in unified_data table)*

---

**Document Owner**: Connor Watson / Claude Code
**Last Updated**: 2025-11-23
**Status**: Experiment running - results pending
