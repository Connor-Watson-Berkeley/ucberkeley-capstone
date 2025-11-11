# Forecast Backtest Results
**Generated**: 2025-11-01 18:03:30
**Evaluation Period**: 42 historical forecast windows (July 2018 - Oct 2024)

---

## Summary

**Best Model (7-day ahead)**: `arima_111_v1`
- MAE: $3.49
- RMSE: $4.56
- MAPE: 3.23%
- Bias: $-0.28

---

## Model Comparison (7-Day Ahead)

| Model | MAE | RMSE | MAPE | Bias | Coverage (95%) |
|-------|-----|------|------|------|----------------|
| arima_111_v1 | $3.49 | $4.56 | 3.23% | $-0.28 | 85.0% |
| sarimax_auto_weather_v1 | $3.55 | $4.58 | 3.27% | $-0.63 | 81.7% |
| xgboost_weather_v1 | $3.77 | $5.12 | 3.49% | $0.02 | 76.7% |
| random_walk_v1 | $4.38 | $5.27 | 4.07% | $-0.40 | 88.3% |
| prophet_v1 | $7.36 | $10.11 | 6.64% | $-1.78 | 53.3% |

---

## Multi-Horizon Performance

### Mean Absolute Error (MAE) by Horizon

| Model | 1-Day | 7-Day | 14-Day |
|-------|-------|-------|--------|
| sarimax_auto_weather_v1 | $1.86 | $3.55 | $4.77 |
| prophet_v1 | $6.35 | $7.36 | $9.28 |
| xgboost_weather_v1 | $1.94 | $3.77 | $5.51 |
| arima_111_v1 | $1.85 | $3.49 | $4.99 |
| random_walk_v1 | $1.94 | $4.38 | $6.17 |

---

## Prediction Interval Calibration

Well-calibrated models should have ~95% coverage for 95% prediction intervals.

| Model | Coverage | Status |
|-------|----------|--------|
| sarimax_auto_weather_v1 | 81.7% | ❌ Poorly calibrated |
| prophet_v1 | 53.3% | ❌ Poorly calibrated |
| xgboost_weather_v1 | 76.7% | ❌ Poorly calibrated |
| arima_111_v1 | 85.0% | ❌ Poorly calibrated |
| random_walk_v1 | 88.3% | ⚠️ Slightly miscalibrated |

---

## Interpretation

### Metrics Explained

- **MAE (Mean Absolute Error)**: Average absolute difference between forecast and actual. Lower is better.
- **RMSE (Root Mean Squared Error)**: Square root of average squared errors. Penalizes large errors more than MAE.
- **MAPE (Mean Absolute Percentage Error)**: MAE as percentage of actual value. Lower is better.
- **Bias**: Average error (forecast - actual). Positive = overforecasting, Negative = underforecasting.
- **Coverage**: % of actuals that fall within 95% prediction interval. Should be ~95% for calibrated models.

### Recommendations

1. **Production Model**: Use `arima_111_v1` for best accuracy
2. **Ensemble Strategy**: Consider averaging top 3 models for robustness
3. **Risk Management**: Use 95% prediction intervals for position sizing (VaR/CVaR)
4. **Model Monitoring**: Track if live forecast errors exceed backtest MAE by >20%
