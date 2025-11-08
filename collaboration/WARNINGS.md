# Warnings & Critical Issues

Last Updated: 2025-11-07

## Active Warnings

### GDELT Data Coverage is Sparse (2025-11-07)

**Severity**: Medium

**Issue**: GDELT data has very limited coverage - only 32 unique days across 3 years:
- 2021: 1 day (Jan 1)
- 2023: 3 days (Sept 30, Oct 8, Oct 16)
- 2025: 28 days (Oct 5 - Nov 2)

**Impact**: GDELT sentiment/news data likely has minimal impact on forecasting models.

**Recommendation**: Consider whether to include GDELT in training data or exclude it entirely.

**Data Issues**:
- Date column is STRING type (`YYYYMMDDHHmmss`) instead of proper DATE
- Missing entire years (2022, 2024)
- Total: 114,221 events across 32 days

---

### Weather v2 Migration in Progress (2025-11-07)

**Severity**: Info

**Status**: Creating bronze table (in progress)

**What's changing**: Migrating from weather (v1) to weather_v2 with corrected growing region coordinates.

**Timeline**:
- Weather backfill: Completed (2025-11-07)
- Bronze table creation: In progress
- unified_data update: Pending
- Model retraining: Pending

**Impact**: New models trained on weather_v2 will have better accuracy due to correct coordinates.

---

## Resolved Warnings

(Archive resolved warnings here)
