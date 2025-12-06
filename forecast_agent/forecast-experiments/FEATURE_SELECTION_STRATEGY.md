# Feature Selection Strategy for Multi-Regional Weather Forecasting

**Date**: 2025-11-23
**Purpose**: Identify most predictive features from 176+ dimensional pivoted weather data
**Status**: PLANNED - Run after multi-regional pivot experiment completes

---

## Motivation

**Current State**: Multi-regional pivot creates **176-201 features**:
- 176 weather features (22 regions × 8 weather vars)
- +1 VIX (optional)
- +24 forex rates (optional)

**Problems**:
1. **Overfitting risk**: 176 features on 3024 training samples (ratio 1:17)
2. **Slow training**: 3-5x slower than single-region models
3. **Interpretability**: Which regions/features actually matter?

**Goal**: Reduce to **top 20-50 features** that capture most predictive power

---

## Research Questions

### 1. Regional Importance
**Question**: Which coffee-growing regions' weather is most predictive?

**Hypothesis**:
- Brazil (40% global production) → High importance
- Vietnam (20% global production) → Medium importance
- Small producers → Low importance

**Test**: Rank regions by cumulative feature importance

### 2. Weather Variable Importance
**Question**: Which weather properties matter most?

**Candidates**:
- `temp_min_c` (frost risk in Brazil)
- `precipitation_mm` (drought risk)
- `temp_mean_c` (growing conditions)
- `humidity_mean_pct` (disease risk)

**Test**: Aggregate feature importance by weather variable type

### 3. Forex Rate Importance
**Question**: Which currency exchange rates are most predictive?

**Hypothesis**:
- BRL/USD (Brazil Real) → Very high (40% of supply)
- VND/USD (Vietnam Dong) → Medium (20% of supply)
- COP/USD (Colombia Peso) → Medium (10% of supply)
- Others → Low importance

**Test**: Rank forex features by importance

---

## Feature Selection Techniques

### Technique 1: Tree-Based Feature Importance

**Method**: Train XGBoost/Random Forest, extract built-in feature importances

**Pros**:
- Fast
- Captures non-linear relationships
- Handles multicollinearity well

**Implementation**:
```python
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

# Train XGBoost
model = XGBRegressor(n_estimators=100, max_depth=5)
model.fit(X_train, y_train)

# Get feature importances
importances = model.feature_importances_
top_features = sorted(zip(feature_names, importances),
                     key=lambda x: x[1], reverse=True)[:50]
```

**Output**: Ranked list of top 50 features

### Technique 2: SHAP Values

**Method**: Use SHAP (SHapley Additive exPlanations) for model-agnostic importance

**Pros**:
- Works with deep learning models (DARTS)
- Provides directional importance (positive/negative)
- More theoretically grounded than tree importance

**Implementation**:
```python
import shap

# Train model
model.fit(X_train, y_train)

# Calculate SHAP values
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Get mean absolute SHAP values
mean_shap = np.abs(shap_values.values).mean(axis=0)
top_features = sorted(zip(feature_names, mean_shap),
                     key=lambda x: x[1], reverse=True)[:50]
```

**Output**: Feature importance with confidence intervals

### Technique 3: Permutation Importance

**Method**: Shuffle each feature, measure impact on validation MAPE

**Pros**:
- Model-agnostic
- Captures true predictive importance
- Less biased than tree importance

**Implementation**:
```python
from sklearn.inspection import permutation_importance

# Train model
model.fit(X_train, y_train)

# Permutation importance
perm_importance = permutation_importance(
    model, X_val, y_val,
    n_repeats=10,
    scoring='neg_mean_absolute_percentage_error'
)

top_features = sorted(zip(feature_names, perm_importance.importances_mean),
                     key=lambda x: x[1], reverse=True)[:50]
```

**Output**: Importance with statistical significance

### Technique 4: Correlation Analysis

**Method**: Measure correlation between each feature and target price

**Pros**:
- Fast
- Identifies linear relationships
- Good for initial screening

**Cons**:
- Misses non-linear relationships
- Ignores feature interactions

**Implementation**:
```python
import pandas as pd

# Calculate correlations
correlations = X_train.corrwith(y_train).abs()
top_features = correlations.nlargest(50)
```

**Output**: Linear correlation coefficients

### Technique 5: Recursive Feature Elimination (RFE)

**Method**: Iteratively remove least important features, retrain, repeat

**Pros**:
- Accounts for feature interactions
- Optimizes for specific target feature count

**Cons**:
- Slow (requires many retraining cycles)
- Can be unstable

**Implementation**:
```python
from sklearn.feature_selection import RFE

# RFE with XGBoost
estimator = XGBRegressor()
selector = RFE(estimator, n_features_to_select=50, step=10)
selector.fit(X_train, y_train)

selected_features = [f for f, s in zip(feature_names, selector.support_) if s]
```

**Output**: Exactly 50 selected features

---

## Recommended Approach: Ensemble Selection

**Strategy**: Combine multiple techniques, select features that rank highly across methods

**Algorithm**:
```python
# Run all 5 techniques
tree_top_50 = get_tree_importance_top_50()
shap_top_50 = get_shap_importance_top_50()
perm_top_50 = get_permutation_importance_top_50()
corr_top_50 = get_correlation_top_50()
rfe_top_50 = get_rfe_top_50()

# Count how many times each feature appears in top 50
vote_counts = Counter()
for feature in tree_top_50 + shap_top_50 + perm_top_50 + corr_top_50 + rfe_top_50:
    vote_counts[feature] += 1

# Select features that appear in at least 3/5 methods
consensus_features = [f for f, count in vote_counts.items() if count >= 3]
```

**Rationale**: Features that rank highly across multiple methods are robustly important

---

## Analysis Dimensions

### By Region

Group feature importances by region to answer: **Which regions matter?**

```python
regional_importance = {}
for region in regions:
    region_features = [f for f in top_features if region in f]
    regional_importance[region] = sum(importance[f] for f in region_features)

# Rank regions
top_regions = sorted(regional_importance.items(),
                    key=lambda x: x[1], reverse=True)
```

**Expected Top 5**:
1. Minas_Gerais_Brazil (largest producer region)
2. Bahia_Brazil
3. Central_Highlands_Vietnam
4. Antioquia_Colombia
5. Sidamo_Ethiopia

### By Weather Variable

Group by weather type to answer: **Which weather properties matter?**

```python
variable_importance = {}
for var in ['temp_min_c', 'temp_max_c', 'precipitation_mm', ...]:
    var_features = [f for f in top_features if f.startswith(var)]
    variable_importance[var] = sum(importance[f] for f in var_features)

# Rank variables
top_variables = sorted(variable_importance.items(),
                      key=lambda x: x[1], reverse=True)
```

**Expected Top 3**:
1. `temp_min_c` (frost risk)
2. `precipitation_mm` (drought risk)
3. `temp_mean_c` (growing conditions)

### By Forex Rate

Rank forex importance to answer: **Which currencies matter?**

```python
forex_importance = {
    forex: importance[forex] for forex in forex_features
}

top_forex = sorted(forex_importance.items(),
                  key=lambda x: x[1], reverse=True)
```

**Expected Top 3**:
1. `cop_usd` (Colombia Peso - coffee exporter)
2. `brl_usd` (Brazil Real - largest producer)
3. `vnd_usd` (Vietnam Dong - 2nd largest producer)

---

## Feature Selection Experiment Design

### Experiment 1: Tree-Based Quick Screen

**Purpose**: Fast initial screening using XGBoost
**Runtime**: ~5 minutes
**Output**: Top 50 features

### Experiment 2: SHAP Deep Dive

**Purpose**: Understand directional importance
**Runtime**: ~30 minutes
**Output**: Feature importance with confidence intervals

### Experiment 3: Permutation Validation

**Purpose**: Validate XGBoost/SHAP findings
**Runtime**: ~20 minutes
**Output**: Statistically significant features

### Experiment 4: Reduced Feature Re-training

**Purpose**: Test if top-50 features match full 176-feature performance

**Test**:
1. Train TCN with all 176 features → baseline MAPE
2. Train TCN with top 50 features → compare MAPE
3. Train TCN with top 30 features → compare MAPE
4. Train TCN with top 20 features → compare MAPE

**Success Criteria**:
- Top 50 achieves ≥95% of full feature performance
- Top 30 achieves ≥90% of full feature performance
- Top 20 achieves ≥80% of full feature performance

**Expected Outcome**: Diminishing returns curve
```
Features | MAPE  | % of Full Performance
---------|-------|---------------------
176      | 1.20% | 100%
50       | 1.23% | 97%
30       | 1.28% | 94%
20       | 1.35% | 89%
10       | 1.50% | 80%
```

---

## Production Integration Plan

### If Feature Selection Succeeds

**Scenario**: Top 50 features achieve >95% of full performance

**Production Pipeline**:
```python
# In forecast_agent/ground_truth/features/

SELECTED_FEATURES = [
    'temp_min_c_minas_gerais_brazil',
    'temp_min_c_bahia_brazil',
    'precipitation_mm_central_highlands_vietnam',
    'cop_usd',
    'brl_usd',
    # ... (50 total)
]

def create_selected_regional_features(df):
    """
    Pivot weather by region, then select only important features.
    """
    # Full pivot (176 features)
    df_pivoted = pivot_regional_weather(df)

    # Select important features only
    df_selected = df_pivoted[SELECTED_FEATURES + ['close']]

    return df_selected
```

**Benefits**:
1. **Faster training**: 50 features vs 176 (3x speedup)
2. **Lower overfitting**: Better feature/sample ratio
3. **Cheaper inference**: Less weather API calls needed
4. **More interpretable**: Can explain which regions/variables drive predictions

---

## Visualization Plan

### 1. Regional Importance Map

**Visual**: World map colored by regional importance

```python
import plotly.express as px

fig = px.choropleth(
    regional_importance_df,
    locations='region',
    color='importance',
    title='Coffee Region Predictive Importance'
)
```

### 2. Feature Importance Heatmap

**Visual**: Heatmap of [regions × weather variables]

```
                temp_min_c  precipitation_mm  humidity_pct  ...
Brazil          ████████    ██████           ███
Vietnam         ████        ████████         ████
Colombia        ███         ████             ██
Ethiopia        ██          ███              ██
```

### 3. Forex Importance Bar Chart

**Visual**: Horizontal bar chart of forex rates

```
cop_usd  ████████████████████  0.85
brl_usd  ████████████████      0.72
vnd_usd  ████████████          0.58
...
```

### 4. Cumulative Importance Curve

**Visual**: Show how much variance explained by top-N features

```
100% |                    ____________
     |              ______/
     |         ____/
     |    ____/
  0% |___/
     0   10   20   30   40   50
         Number of Features
```

**Interpretation**: Steep curve → few features explain most variance (good for reduction)

---

## Expected Findings

### Regional Ranking (Predicted)

| Rank | Region | Cumulative Importance | % Global Production |
|------|--------|--------------------|-------------------|
| 1 | Minas_Gerais_Brazil | 35% | 25% |
| 2 | Central_Highlands_Vietnam | 25% | 15% |
| 3 | Bahia_Brazil | 15% | 15% |
| 4 | Antioquia_Colombia | 10% | 8% |
| 5 | Sidamo_Ethiopia | 5% | 4% |
| 6-22 | Others | 10% | 33% |

**Insight**: Top 5 regions explain 90% of predictive power

### Weather Variable Ranking (Predicted)

| Rank | Variable | Importance | Why? |
|------|----------|-----------|------|
| 1 | temp_min_c | 40% | Frost risk (Brazil) |
| 2 | precipitation_mm | 30% | Drought risk |
| 3 | temp_mean_c | 15% | Growing conditions |
| 4 | humidity_mean_pct | 8% | Disease risk |
| 5 | rain_mm | 4% | (Correlated with precip) |
| 6-8 | Others | 3% | Minimal impact |

**Insight**: Top 3 weather vars explain 85% of weather signal

### Forex Ranking (Predicted)

| Rank | Currency | Importance | Country |
|------|----------|-----------|---------|
| 1 | cop_usd | 35% | Colombia |
| 2 | brl_usd | 30% | Brazil |
| 3 | vnd_usd | 20% | Vietnam |
| 4 | etb_usd | 8% | Ethiopia |
| 5 | hnl_usd | 4% | Honduras |
| 6-24 | Others | 3% | Minor producers |

**Insight**: Top 3 currencies explain 85% of forex signal

---

## Timeline

**After Multi-Regional Pivot Completes**:
1. **Day 1**: Run all 5 feature selection techniques (parallel)
2. **Day 1**: Ensemble voting, select top 50 consensus features
3. **Day 2**: Regional/variable/forex analysis
4. **Day 2**: Create visualizations
5. **Day 3**: Re-train models with selected features, compare performance
6. **Day 4**: Document findings, update feature engineering pipeline

**Total Duration**: 4 days

---

## Success Metrics

### Primary Metric: Performance Retention

**Target**: Top 50 features achieve ≥95% of full 176-feature performance

**Tiers**:
- **Excellent**: Top 30 features ≥95% performance
- **Good**: Top 50 features ≥95% performance
- **Acceptable**: Top 50 features ≥90% performance
- **Needs work**: Top 50 features <90% performance

### Secondary Metrics

1. **Training Speedup**: 2-3x faster with selected features
2. **Regional Concentration**: Top 5 regions explain ≥80% of importance
3. **Variable Concentration**: Top 3 weather vars explain ≥70% of importance
4. **Forex Concentration**: Top 3 currencies explain ≥70% of importance

---

## Related Documents

- **MULTI_REGIONAL_PIVOT_EXPERIMENT.md**: Describes 176-feature pivot experiment
- **WEATHER_PREDICTIVE_POWER_FINDING.md**: Shows weather is highly predictive
- **forecast_agent/ground_truth/features/data_preparation.py**: Where to implement selection

---

## Open Questions

1. **Should we use different features for different horizons?**
   - 1-day: Regional specificity matters more
   - 14-day: Global trends matter more
   - Possible: Adaptive feature selection by horizon

2. **Should we include lagged features?**
   - Example: temp_min_c_brazil[t-1], temp_min_c_brazil[t-7]
   - Captures weather momentum/trends
   - Risk: Adds another dimension to feature space

3. **Should we engineer interaction features?**
   - Example: temp_min_c_brazil × precipitation_mm_brazil
   - Captures compound weather events (cold + dry = frost risk)
   - Risk: Combinatorial explosion

4. **Should we use PCA instead of feature selection?**
   - Pros: Captures linear combinations, reduces multicollinearity
   - Cons: Less interpretable, loses regional/variable identity

---

**Document Owner**: Connor Watson / Claude Code
**Last Updated**: 2025-11-23
**Status**: Planned - waiting for pivot experiment completion
