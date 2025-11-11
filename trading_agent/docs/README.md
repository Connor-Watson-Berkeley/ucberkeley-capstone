# Trading Agent Technical Documentation

This folder contains technical documentation for implementation details, analyses, and archived results.

---

## Contents

### Analysis Documentation

**[ACCURACY_THRESHOLD_ANALYSIS.md](ACCURACY_THRESHOLD_ANALYSIS.md)**
- Synthetic prediction generation methodology
- Accuracy threshold analysis (50%-100%)
- Key finding: 70% minimum accuracy needed for profitability
- Comparison of real models to synthetic benchmarks

**[MULTI_MODEL_MODIFICATIONS.md](MULTI_MODEL_MODIFICATIONS.md)**
- Changes made to create multi-model notebook
- Unity Catalog integration details
- Nested loop structure (commodity → model)
- Result storage format

**[DATA_FORMAT_VERIFICATION.md](DATA_FORMAT_VERIFICATION.md)**
- Data format compatibility testing
- Verification that `forecast_loader.py` produces correct format
- Example data structures and transformations

### Historical Results

**[backtest_results.md](backtest_results.md)**
- Archived backtest results from 42 historical windows
- Model performance comparison (MAE, RMSE, MAPE)
- Legacy results - superseded by multi-model analysis

---

## For Users

If you're looking for user-facing documentation:
- **Daily Recommendations**: [`../operations/README.md`](../operations/README.md)
- **Data Access Guide**: [`../FORECAST_API_GUIDE.md`](../FORECAST_API_GUIDE.md)
- **Main README**: [`../README.md`](../README.md)

---

Last Updated: 2025-11-10
