# Weather Coordinates Issue - CRITICAL DATA QUALITY PROBLEM

**Date**: 2025-11-05
**Status**: 🚨 CRITICAL - Weather data not from actual growing regions

## The Problem

The current `commodity.bronze.weather` table uses coordinates that are **NOT from the actual coffee/sugar growing regions**. Instead, they appear to use:
- State/country capitals
- Regional centroids
- Broad geographic averages

This means the weather data **does not capture local weather events that affect commodity production**.

## Proof: July 2021 Brazil Frost

### What Actually Happened:
- **Date**: July 20-21, 2021
- **Location**: Sul de Minas coffee region, Brazil
- **Temperature**: Dropped to **-2°C to -4°C** (severe frost)
- **Impact**: Destroyed 20-30% of Brazil's coffee crop
- **Market Impact**: Coffee prices spiked 70% (from $130 to $220/lb)

### What Our Data Shows:
```
Minas Gerais Weather - July 20-21, 2021:
- July 20: Min 11.1°C (NO frost detected)
- July 21: Min 14.8°C (NO frost detected)
```

**The weather data completely missed the most significant weather event affecting coffee prices in the past 5 years.**

## Coordinate Comparison

### Minas Gerais, Brazil (Coffee):

| Source | Latitude | Longitude | Location | Distance from Growing Region |
|--------|----------|-----------|----------|------------------------------|
| **Current Lambda** | -18.5122 | -44.5550 | Near Belo Horizonte (state capital) | ~200 km north |
| **Correct (region_coordinates.json)** | -20.3155 | -45.4108 | Sul de Minas (actual coffee region) | ✅ Correct |

### Potential Issues Across All Regions:

Comparing ALL 67 regions:

| Region | Lat Diff | Lon Diff | Est Distance | Likely Issue |
|--------|----------|----------|--------------|--------------|
| Minas_Gerais_Brazil | 1.8° | 0.86° | ~200 km | Capital vs growing region |
| Sao_Paulo_Brazil | 1.34° | 2.11° | ~250 km | Capital vs interior coffee zones |
| Huila_Colombia | 0.61° | 0.25° | ~70 km | Broader region vs specific valleys |
| ... | ... | ... | ... | ... |

*Full comparison needed but initial spot checks show 5-250km discrepancies.*

## Impact on Forecasting

**This fundamentally undermines the predictive value of weather features:**

1. **Local extreme events missed**: Frosts, droughts, heatwaves that occur in specific growing valleys but not in nearby cities
2. **Microclimate differences**: Coffee grows at specific elevations/conditions different from regional averages
3. **False negatives**: Major production-affecting events don't appear in the data
4. **Wrong seasonal patterns**: Capitals may have different rainfall/temperature patterns than agricultural zones

**Result**: Weather features in SARIMAX models likely have weak/no predictive power because they don't measure the actual conditions affecting crops.

## Solution Options

### Option A: Regenerate ALL Historical Weather Data (RECOMMENDED)

**Action**: Re-fetch all historical weather (2015-2025) using correct coordinates

**Steps**:
1. ✅ Already have correct coordinates in `region_coordinates.json`
2. Create backfill script using Open-Meteo with correct lat/lon
3. Write to new S3 prefix: `landing/weather_corrected/`
4. Create new bronze table: `commodity.bronze.weather_v2`
5. Test against July 2021 frost (should now show temps <0°C)
6. Update unified_data to use v2

**Time**: 6-12 hours (67 regions × 3,800 days, rate-limited API)

**Cost**: FREE (Open-Meteo allows historical weather requests)

**Pros**:
- ✅ Fixes the root cause
- ✅ Real weather data from actual growing regions
- ✅ Can validate against known events (2021 frost)
- ✅ Provides proper foundation for forecasting

**Cons**:
- ⏳ Takes time to regenerate
- 📦 Additional S3 storage (~1 GB)

### Option B: Keep Current Data (NOT RECOMMENDED)

**Why not**:
- ❌ Weather features will have minimal/no predictive value
- ❌ Models won't capture critical production-affecting events
- ❌ Cannot properly validate against historical commodity price spikes

## Recommendation

**✅ REGENERATE ALL WEATHER DATA WITH CORRECT COORDINATES**

This is critical for:
1. **SARIMAX with weather features** - Your primary model for improvement
2. **Weather forecast integration** - Forecasts need to match actuals' coordinates
3. **Model validation** - Need to explain historical price movements

The weather forecast system we're building will use the correct coordinates, which means:
- Historical actuals (wrong coords) won't match forecasts (right coords)
- Models trained on synthetic forecasts (correct coords) won't align with training actuals (wrong coords)
- Cannot properly evaluate forecast utility

## Next Steps

1. **Immediate**: Stop synthetic weather forecast backfill until weather actuals are fixed
2. **Create**: Backfill script for historical weather with correct coordinates
3. **Run**: Historical weather backfill (2015-2025) - 6-12 hours
4. **Validate**: Check July 2021 Minas Gerais shows frost (temps <0°C)
5. **Deploy**: Update unified_data to use corrected weather (v2)
6. **Then Resume**: Weather forecast integration with confidence that coords align

## Historical Validation Points

Once regenerated, validate against these known events:

| Date | Event | Expected Signal | Region |
|------|-------|-----------------|--------|
| July 2021 | Brazil frost | Temps -2 to -4°C | Minas Gerais |
| 2015-2016 | El Niño drought | Low rainfall | Vietnam, Indonesia |
| Jan 2019 | Brazil heatwave | Temps >35°C sustained | Sao Paulo/Minas |
| May 2020 | Colombian floods | Heavy rainfall | Huila, Eje Cafetero |

---

**Status**: Awaiting decision to proceed with weather data regeneration
