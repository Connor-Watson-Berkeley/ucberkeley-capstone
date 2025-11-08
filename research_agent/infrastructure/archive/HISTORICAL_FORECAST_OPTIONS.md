# Historical Weather Forecast Options - Complete Analysis

## The Challenge
We need **historical forecasts** (what was predicted on date X for future date Y), NOT historical actuals (what actually happened). This is critical to avoid data leakage.

---

## Free Options

### 1. Open-Meteo Previous Model Runs API
**URL**: https://open-meteo.com/en/docs/previous-runs-api

**Coverage**:
- GFS Temperature: March 2021 onwards ⚠️ **Temperature only, no precip/humidity**
- Most other models: January 2024 onwards
- JMA models: 2018 onwards (Japan/Asia focus)

**Limitations**:
- ⚠️ GFS from 2021 has ONLY temperature, not precipitation/humidity
- ⚠️ Full variable set only from Jan 2024+
- ⚠️ Designed for short-term comparisons (yesterday's forecast for today)
- ⚠️ Not clear if it provides 14-day ahead forecasts from 2021

**Cost**: FREE for non-commercial

**Verdict**: ❓ **Worth Testing** - unclear if it provides the 14-day forecast horizon we need from 2021

---

### 2. NOAA NOMADS Archive (GFS/GEFS)
**URL**: https://nomads.ncep.noaa.gov/

**Coverage**:
- GFS operational forecasts archived back to 2015+
- Full forecast horizon (out to 16 days)
- All variables (temp, precip, humidity, wind)

**Limitations**:
- ⚠️ **GRIB2 format** - complex binary format, requires pygrib/cfgrib
- ⚠️ Large file sizes (~100MB per model run)
- ⚠️ Significant processing/storage overhead
- ⚠️ No simple REST API
- ⚠️ Would need ~1TB storage for full 2021-2025 archive

**Cost**: FREE but high engineering cost

**Verdict**: ❌ **Not Recommended** - too complex for the value

---

## Paid Options

### 3. Visual Crossing Weather API
**URL**: https://www.visualcrossing.com/

**Pricing**: $0.0001 per record
- 1000 records/day FREE tier
- Beyond that: $0.0001/record

**Coverage**: Claims "historical data for past 50 years"

**Critical Unknown**: ⚠️ **Do they provide historical FORECASTS or just historical ACTUALS?**
- Most weather APIs only provide actuals
- Their documentation doesn't clearly state forecast archives
- Need to contact them to verify

**Cost Estimate** (if they have forecast data):
```
67 regions × 1,825 days (2021-2026) × 16 forecast days = 1,954,800 records
Cost: ~$195 for full backfill
Ongoing: 67 regions × 16 days = 1,072 records/day (within free tier!)
```

**Verdict**: ❓ **Contact them first** - need to verify they have forecast archives, not just actuals

---

### 4. Meteomatics Weather API
**URL**: https://www.meteomatics.com/

**Pricing**: Custom enterprise pricing (starts ~$500-1000/month)

**Coverage**: Claims historical forecast data available

**Verdict**: ❌ **Too Expensive** - not worth it for student project

---

### 5. ECMWF MARS Archive
**URL**: https://www.ecmwf.int/

**Pricing**:
- Free for national met services
- License required for others (cost unclear)
- Academic license may be available

**Coverage**: Comprehensive forecast archives from 1979+

**Verdict**: ❓ **Academic License?** - worth checking if UCB has institutional access

---

## Practical Recommendations

### Option A: Open-Meteo Previous Runs (Test First) ✅ RECOMMENDED
**Action**: Test if the API provides 14-day ahead forecasts from March 2021

**Test Script**:
```python
# Try to get: "What did GFS predict on 2021-03-01 for 2021-03-15?"
import requests

url = "https://previous-runs-api.open-meteo.com/v1/forecast"
params = {
    "latitude": 14.56,
    "longitude": -90.73,
    "start_date": "2021-03-01",
    "end_date": "2021-03-15",
    "forecast_days": 14,
    "previous_days": 14  # Look back 14 days
}

response = requests.get(url, params=params)
# If this works, we're golden!
```

**If it works**:
- ✅ FREE
- ✅ No data leakage
- ✅ Temperature from 2021, full variables from 2024
- ⚠️ Missing precip/humidity for 2021-2024

**Cost**: $0

---

### Option B: Visual Crossing (If confirmed) 💰 AFFORDABLE
**Action**: Contact Visual Crossing to verify forecast archive availability

**Questions to ask**:
1. Do you provide historical forecast data (not just historical actuals)?
2. Can I query "what was forecasted on 2021-03-01 for 2021-03-15"?
3. What's the date range for forecast archives?
4. What variables are available (temp, precip, humidity)?

**If confirmed**:
- ✅ Full historical coverage
- ✅ All variables
- ✅ Simple API
- 💰 ~$195 one-time + free ongoing

**Cost**: ~$195 (one-time backfill)

---

### Option C: Start Collecting Now, No Historical 🕐 PATIENT
**Action**: Deploy forecast Lambda today, accumulate data forward

**Timeline**:
- Today: Start collecting
- Month 3: Enough for initial experiments
- Month 6: Enough for robust models

**Pros**:
- ✅ FREE
- ✅ Real forecast data
- ✅ No data leakage

**Cons**:
- ⏳ Must wait 3-6 months
- ❌ No 2021-2025 historical forecasts
- ❌ Can't use for this semester's project

**Cost**: $0

---

### Option D: UCB Academic ECMWF Access 🎓 CHECK
**Action**: Contact UCB library/research computing

**Questions**:
1. Does UCB have institutional access to ECMWF data?
2. Can I access MARS archive for academic research?
3. What's the process to get credentials?

**If available**:
- ✅ FREE (via university)
- ✅ Comprehensive coverage
- ✅ High quality

**Cost**: $0 (if UCB has access)

---

## My Recommendation

### Immediate Action Plan:

1. **Test Open-Meteo Previous Runs API** (30 min)
   - See if it provides 14-day forecasts from 2021
   - If yes: Use it! (even if temperature-only for 2021-2024)

2. **Contact Visual Crossing** (1 hour)
   - Verify forecast archive availability
   - If yes and ~$200: Consider purchasing

3. **Check UCB ECMWF Access** (1 day)
   - Email research computing
   - Could unlock gold mine of data

4. **Deploy Daily Collector** (regardless)
   - Start building real forecast history NOW
   - Takes 1-2 hours to deploy

### Best Case Scenario:
- Open-Meteo works → FREE, immediate, temperature-only 2021-2024
- Deploy daily collector → accumulate full variables going forward
- **Total cost: $0**

### Fallback:
- Visual Crossing has forecasts → $195 one-time
- Deploy daily collector → free ongoing
- **Total cost: $195 one-time**

### Worst Case:
- Nothing works for historical
- Deploy daily collector → wait 3-6 months
- **Can't use forecasts this semester**

---

## Next Steps

Should I:
1. **Test Open-Meteo Previous Runs API** to see if it meets our needs?
2. **Draft email to Visual Crossing** to inquire about forecast archives?
3. **Just deploy the daily collector** and defer historical forecasts?

Your call!
