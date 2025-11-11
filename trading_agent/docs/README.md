# Trading Agent Technical Documentation

This folder contains technical documentation for implementation details and analyses.

---

## Contents

### Multi-Model Analysis

**[MULTI_MODEL_ANALYSIS.md](MULTI_MODEL_ANALYSIS.md)**

Comprehensive guide to the multi-model backtesting framework:
- **Implementation details**: Unity Catalog connection, model discovery, data loading
- **Synthetic predictions**: How accuracy-controlled forecasts are generated
- **Accuracy threshold analysis**: Finding that 70% accuracy is minimum for profitability
- **Real model benchmarking**: Comparison to synthetic accuracy levels
- **Usage examples**: Running analysis, comparing models, interpreting results
- **Statistical insights**: Performance vs accuracy, diminishing returns, ceiling analysis

**Key Finding:** Best real model (`sarimax_auto_weather_v1`) performs at ~75% effective accuracy, providing $2,390 advantage over baseline strategies.

---

## For Users

If you're looking for user-facing documentation:
- **Quick Start & Overview**: [`../README.md`](../README.md)
- **Daily Recommendations**: [`../operations/README.md`](../operations/README.md)
- **Unity Catalog Queries**: [`../FORECAST_API_GUIDE.md`](../FORECAST_API_GUIDE.md)

---

## Archived Documentation

Previous versions of technical documentation have been consolidated:
- ~~MULTI_MODEL_MODIFICATIONS.md~~ → Merged into MULTI_MODEL_ANALYSIS.md
- ~~ACCURACY_THRESHOLD_ANALYSIS.md~~ → Merged into MULTI_MODEL_ANALYSIS.md
- ~~DATA_FORMAT_VERIFICATION.md~~ → Removed (historical verification, no longer needed)
- ~~backtest_results.md~~ → Removed (outdated Nov 1 results, superseded by multi-model analysis)

---

Last Updated: 2025-11-10
