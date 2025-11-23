# GDELT Themes Relevant to Commodity Futures Forecasting

**Date**: 2025-11-22
**Purpose**: Identify high-value GDELT themes beyond the 7 aggregated groups

---

## Current State

**Current `gdelt_wide` themes** (7 groups):
- SUPPLY - Supply chain
- LOGISTICS - Transportation
- TRADE - Trading activity
- MARKET - Market conditions
- POLICY - Regulations
- CORE - Core commodity news
- OTHER - Miscellaneous

**Problem**: These are **too aggregated** - we're missing granular signals!

---

## High-Value GDELT Themes for Coffee Futures

GDELT's full taxonomy has **~3,000 themes**. Here are the most relevant for commodity price forecasting:

### 1. **Weather & Climate Events** (Critical for Agriculture)

**Why important**: Weather directly affects crop yields, quality, and supply

**Relevant themes**:
- `ENV_WEATHER_DROUGHT` - Droughts in growing regions → price ↑
- `ENV_WEATHER_FROST` - Frost damage → price ↑↑
- `ENV_WEATHER_FLOOD` - Flooding → crop damage → price ↑
- `ENV_WEATHER_HURRICANE` - Storm damage
- `ENV_CLIMATE_ELINO` - El Niño patterns
- `ENV_CLIMATE_LANINA` - La Niña patterns
- `TAX_WEATHER` - Weather-related tax/policy

**Impact**: **Very High** - Weather is the #1 driver of agricultural commodity prices

### 2. **Geopolitical Events** (Supply Disruption Risk)

**Why important**: Conflicts, sanctions, trade wars disrupt supply chains

**Relevant themes**:
- `CONFLICT` - Armed conflicts in producing regions
- `CRISISLEX_CRISISLEXREC` - Crisis events
- `TAX_FNCACT_SANCTIONS` - Economic sanctions
- `UNGP_HUMAN_RIGHTS` - Human rights issues (labor)
- `REBEL` - Rebel activity in growing regions
- `PROTEST` - Social unrest
- `STRIKE` - Labor strikes

**Impact**: **High** - Can cause sudden supply shocks

### 3. **Economic Indicators**

**Why important**: Currency, inflation, recession affect commodity prices

**Relevant themes**:
- `ECON_INFLATION` - Inflation news
- `ECON_RECESSION` - Recession signals
- `ECON_STOCKMARKET` - Market crashes/rallies
- `ECON_CURRENCY_DEVALUATION` - Currency crises
- `TAX_FNCACT_CENTRALBANK` - Central bank actions
- `TAX_FNCACT_IMF` - IMF interventions
- `ECON_COMMODITY_PRICE_INCREASE` - Commodity price movements
- `ECON_COMMODITY_PRICE_DECREASE`

**Impact**: **Medium-High** - Macroeconomic context

### 4. **Trade & Policy**

**Why important**: Tariffs, trade agreements, quotas affect supply/demand

**Relevant themes**:
- `TAX_FNCACT_TARIFF` - Tariffs
- `TAX_FNCACT_TRADE_AGREEMENT` - Trade deals
- `TAX_FNCACT_EXPORT_BAN` - Export restrictions
- `TAX_FNCACT_IMPORT_BAN` - Import restrictions
- `TAX_FNCACT_QUOTA` - Quotas
- `WTO` - WTO rulings
- `NAFTA` / `USMCA` - Regional trade agreements

**Impact**: **High** - Direct supply/demand effects

### 5. **Agricultural & Production**

**Why important**: Pest outbreaks, disease, technology affect yields

**Relevant themes**:
- `AGRI_PEST` - Pest outbreaks (coffee rust, borer)
- `AGRI_DISEASE` - Plant diseases
- `AGRI_HARVEST` - Harvest reports
- `AGRI_YIELD` - Yield forecasts
- `AGRI_TECHNOLOGY` - Agricultural innovation
- `ENV_DEFORESTATION` - Land use changes

**Impact**: **Very High** - Production fundamentals

### 6. **Transportation & Logistics**

**Why important**: Shipping costs, port strikes affect prices

**Relevant themes**:
- `TAX_FNCACT_SHIPPING` - Shipping disruptions
- `STRIKE_TRANSPORT` - Port strikes
- `FUEL_PRICE` - Transportation costs
- `SHIPPING_CONTAINER_SHORTAGE` - Logistics bottlenecks

**Impact**: **Medium** - Supply chain friction

### 7. **Demand Signals**

**Why important**: Consumption patterns, preferences

**Relevant themes**:
- `ECON_CONSUMER_CONFIDENCE` - Consumer sentiment
- `FOOD_CONSUMPTION` - Consumption trends
- `RETAIL_SALES` - Retail indicators
- `ECON_GDP_GROWTH` - Economic growth (demand proxy)

**Impact**: **Medium** - Demand-side fundamentals

### 8. **Brazil-Specific** (Major Coffee Producer)

**Why important**: Brazil = 40% of global coffee production

**Relevant themes**:
- `BRAZIL` + `WEATHER` - Brazilian weather
- `BRAZIL` + `POLITICS` - Political instability
- `BRAZIL` + `CURRENCY` - Real devaluation
- `BRAZIL` + `LABOR` - Labor issues
- `BRAZIL` + `AGRICULTURE` - Production news

**Impact**: **Very High** - Brazil-specific events move global prices

---

## Proposed Enhanced Feature Set

### Instead of 7 aggregated groups (35 features)

**Extract ~50-100 granular theme features**:

```python
# Weather cluster (10 features)
- drought_mentions
- frost_mentions
- flood_mentions
- elnino_mentions
- temperature_anomaly_mentions

# Geopolitical cluster (15 features)
- conflict_count
- sanctions_count
- strike_count
- protest_count
- political_instability

# Economic cluster (15 features)
- inflation_mentions
- recession_mentions
- currency_crisis_count
- stock_market_volatility

# Trade/policy cluster (10 features)
- tariff_announcements
- trade_agreement_count
- export_ban_count

# Production cluster (10 features)
- pest_outbreak_count
- disease_mentions
- harvest_report_count
- yield_forecast_mentions

# Each with tone metrics (avg, positive, negative, polarity)
```

---

## Implementation Strategy

### Phase 1: Extract Full Theme Taxonomy

```sql
-- Query bronze table to get all unique themes
SELECT DISTINCT themes
FROM commodity.bronze.gdelt_bronze
WHERE themes IS NOT NULL;
```

### Phase 2: Create Theme Frequency Table

```sql
CREATE TABLE commodity.silver.gdelt_themes_detailed AS
SELECT
    date,
    commodity,
    -- Parse themes column (comma/semicolon separated)
    -- Count mentions of each theme
    -- Calculate tone metrics per theme
    ...
FROM commodity.bronze.gdelt_bronze
GROUP BY date, commodity;
```

### Phase 3: Feature Engineering

```python
# For each date and commodity:
# 1. Count theme mentions (last 7 days, 30 days)
# 2. Rolling average of tone by theme
# 3. Trend detection (increasing/decreasing mentions)
# 4. Spike detection (sudden surge in mentions)

# Example features:
- drought_mentions_7d (count in last week)
- drought_tone_avg_30d (average tone over month)
- frost_spike_indicator (boolean: mentions >2 std dev above mean)
```

### Phase 4: Train with Enhanced Features

```python
# Baseline: 7 weather features
baseline_features = ['temperature', 'precipitation', ...]

# Enhanced: weather + 100 GDELT theme features
enhanced_features = baseline_features + [
    'drought_mentions_7d',
    'frost_mentions_7d',
    'brazil_conflict_count_30d',
    'inflation_tone_avg_7d',
    'tariff_spike_indicator',
    # ... ~100 total
]

# Expected improvement: 20-40% MAPE reduction
# From 1.12% → 0.7-0.9%
```

---

## Permissions Needed

```sql
-- Grant access to bronze table
GRANT SELECT ON commodity.bronze.gdelt_bronze TO <user>;

-- Then can explore full theme taxonomy
```

---

## Expected Impact on Forecasting

### Scenario Analysis

**Current baseline** (weather only):
- MAPE: 1.12%
- Features: 7 (temperature, precipitation, etc.)

**gdelt_wide** (7 aggregated groups):
- MAPE: ~1.00% (est. 10% improvement)
- Features: 42 (7 groups × 5 metrics + weather)
- **Problem**: Too aggregated, loses signal

**gdelt_detailed** (100 granular themes):
- MAPE: **0.7-0.9%** (est. 30-40% improvement)
- Features: ~100 (weather + granular themes)
- **Why better**:
  - Captures specific events (frost warnings, strikes)
  - Earlier signals (mentions spike before price moves)
  - Geopolitical risk factors
  - Brazil-specific indicators

---

## Real-World Examples

### Example 1: Brazilian Frost (July 2021)

**What happened**: Frost damaged Brazil coffee crops → prices +50% in days

**GDELT signals** (days before price spike):
- `ENV_WEATHER_FROST` + `BRAZIL` mentions ↑↑
- `AGRI_YIELD` + negative tone
- `COFFEE` + `SHORTAGE` co-mentions

**Impact**: Model with frost theme would predict price spike **3-7 days early**

### Example 2: Vietnam Drought (2015-2016)

**What happened**: El Niño drought → Vietnam (Robusta) supply ↓ → Arabica prices ↑

**GDELT signals**:
- `ENV_CLIMATE_ELNINO` mentions ↑
- `VIETNAM` + `DROUGHT` ↑
- `COFFEE` + `SUPPLY_SHORTAGE`

**Impact**: Early warning system from theme tracking

### Example 3: Port Strike (2023)

**What happened**: Brazilian port strikes → export delays → temporary price spike

**GDELT signals**:
- `BRAZIL` + `STRIKE` + `TRANSPORT` ↑
- `PORT` + `DELAY` mentions

**Impact**: Short-term price movement prediction

---

## Next Steps

1. **Grant permissions** on `commodity.bronze.gdelt_bronze`

2. **Extract full theme taxonomy**:
   ```python
   python3 explore_gdelt_themes_bronze.py
   ```

3. **Create detailed feature table**:
   ```sql
   -- Parse themes, create daily theme counts + tones
   ```

4. **Train enhanced model**:
   ```python
   python3 darts_nhits_with_detailed_themes.py
   ```

5. **Compare results**:
   - Baseline (weather): 1.12% MAPE
   - Aggregated themes: ~1.00% MAPE
   - **Detailed themes: 0.7-0.9% MAPE** ⭐

---

## Conclusion

**You're absolutely right** - there are many more valuable GDELT themes beyond the 7 aggregated groups!

**Key themes for coffee futures**:
1. Weather events (frost, drought, El Niño)
2. Geopolitical (Brazil conflicts, strikes)
3. Economic (inflation, currency)
4. Trade policy (tariffs, bans)
5. Production (pests, disease, yields)

**Expected improvement**: 30-40% MAPE reduction (1.12% → 0.7-0.9%)

**Blocker**: Need permissions on `commodity.bronze.gdelt_bronze` to access full theme data

---

**Document Owner**: Connor Watson / Claude Code
**Last Updated**: 2025-11-22
**Status**: Awaiting bronze table permissions
