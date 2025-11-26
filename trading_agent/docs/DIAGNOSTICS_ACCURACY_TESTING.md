# Accuracy Level Testing Guide

**Purpose:** Test trading strategies across different prediction accuracy levels to validate algorithms and diagnose issues

**Created:** 2025-11-24
**Status:** Ready for v8 predictions

---

## Testing Philosophy

### The Accuracy Spectrum Test

We test with 5 accuracy levels to understand algorithm behavior:

```
100% (Perfect Foresight) → Algorithm validation (algorithms MUST work)
90% → Performance validation (predictions should add value)
80% →
70% → Monotonicity validation (performance should improve with accuracy)
60% →
```

---

## Test 1: Algorithm Validation (100% Accuracy) - CRITICAL

### Purpose
**Prove the trading algorithms work correctly**

### Logic
With 100% accurate predictions (perfect foresight):
- Prediction strategies **MUST** beat baseline strategies
- If they don't, the algorithms are **fundamentally broken**

This is NOT a test of prediction quality. This is a test of algorithm correctness.

### Expected Results
```
Coffee synthetic_acc100:
- Best Baseline (Equal Batches): ~$727k
- Best Prediction (Expected Value): >$800k (+10% minimum)
```

### How to Run
```bash
cd diagnostics/
python diagnostic_100_algorithm_validation.py
```

### Interpretation

**✓ PASS (Predictions beat baselines by >10%):**
- Algorithms work correctly
- Any underperformance with real predictions is due to:
  - Prediction accuracy not high enough
  - Parameter tuning needed
  - Prediction usage needs refinement

**❌ FAIL (Predictions lose to baselines even with 100% accuracy):**
- **CRITICAL BUG** in algorithm logic
- Stop all other tests and debug immediately
- Possible issues:
  - Decision logic inverted (buy when should sell)
  - Wrong prediction horizon used
  - Cost calculations broken
  - Predictions not being looked up correctly
- Next step: Run `diagnostic_17_paradox_analysis.ipynb`

---

## Test 2: Monotonicity Validation (60%, 70%, 80%, 90%, 100%)

### Purpose
Verify that performance improves as prediction accuracy increases

### Expected Behavior
```
Performance should follow this pattern:

100% accuracy: >$800k  ┐
 90% accuracy:  $775k  │ Monotonic
 80% accuracy:  $760k  │ Improvement
 70% accuracy:  $740k  │
 60% accuracy:  $730k  ┘
Baseline:       $727k
```

### How to Run
This is tested in the main notebooks (05 and 11) by comparing results across all accuracy levels.

### Interpretation

**✓ PASS (Monotonic improvement):**
- Algorithms respond correctly to prediction quality
- System is working as designed
- Can focus on improving prediction accuracy

**❌ FAIL (Random or declining pattern):**
- Algorithm is NOT using predictions correctly
- Possible issues:
  - Predictions passed but not used in decisions
  - Decision thresholds too conservative/aggressive
  - Wrong confidence calculations
- Next step: Run Phase 2 diagnostics (11-14) from test plan

---

## Test 3: Performance Validation (90% Accuracy)

### Purpose
Validate that high-quality predictions (90% accuracy) meaningfully improve results

### Expected Behavior
```
90% accuracy predictions should:
- Beat best baseline by +$30k to $50k (+4% to +7%)
- Show prediction-informed decisions (not fallback trades)
- Make confident trades when predictions are strong
```

### Actual Behavior (Current Problem)
```
Coffee synthetic_acc90:
- Best Baseline: $727,037
- Best Prediction: $708,017
- Result: -$19,020 (-2.6%) ❌ PREDICTION PARADOX
```

### How to Run
Main notebooks (05 and 11) automatically test this when synthetic_acc90 is available.

### Interpretation

**✓ PASS (Predictions beat baseline by >$30k):**
- System is working correctly
- 90% accuracy is sufficient for value-add
- Focus on production deployment

**❌ FAIL (Predictions underperform):**
- If 100% test passed: Issue is with 90% accuracy level (not algorithms)
- If 100% test failed: Fix algorithms first, then retest
- Diagnostics to run:
  1. Check if predictions are being passed (diagnostic 08) ✓ COMPLETED
  2. Trace decision logic (diagnostic 11-14)
  3. Analyze trade-by-trade differences (diagnostic 17)

---

## Test 4: Threshold Identification (60%-90%)

### Purpose
Identify the minimum prediction accuracy needed for value-add

### Expected Behavior
```
Accuracy threshold for break-even: ~70%
- Below 70%: Predictions don't add value (noise dominates)
- Above 70%: Predictions provide actionable signal
```

### How to Run
Compare results from 60%, 70%, 80%, 90% in notebook 11:
```python
# Plot: Net Earnings vs Accuracy
# Find where prediction line crosses baseline line
```

### Interpretation
The accuracy threshold tells us:
- **Threshold = 60%**: Algorithms very efficient (can use weak predictions)
- **Threshold = 70%**: Expected (moderate accuracy needed)
- **Threshold = 90%**: Problem (algorithms too conservative)

---

## Testing Workflow

### Step 1: Wait for v8 Predictions
```bash
# Check if v8 is complete
databricks fs ls dbfs:/Volumes/commodity/trading_agent/files/ | grep validation_results_v8

# Download predictions
databricks fs cp dbfs:/Volumes/commodity/trading_agent/files/prediction_matrices_coffee_synthetic_acc100_v8.pkl ./
databricks fs cp dbfs:/Volumes/commodity/trading_agent/files/prediction_matrices_coffee_synthetic_acc90_v8.pkl ./
databricks fs cp dbfs:/Volumes/commodity/trading_agent/files/prediction_matrices_coffee_synthetic_acc80_v8.pkl ./
databricks fs cp dbfs:/Volumes/commodity/trading_agent/files/prediction_matrices_coffee_synthetic_acc70_v8.pkl ./
databricks fs cp dbfs:/Volumes/commodity/trading_agent/files/prediction_matrices_coffee_synthetic_acc60_v8.pkl ./
```

### Step 2: Run Algorithm Validation (CRITICAL)
```bash
cd diagnostics/
python diagnostic_100_algorithm_validation.py
```

**STOP if this fails!** Fix algorithms before continuing.

### Step 3: Run Full Backtest Suite
In Databricks, run notebooks:
1. `05_strategy_comparison.ipynb` - Auto-discovers all accuracy levels
2. `11_synthetic_accuracy_comparison.ipynb` - Compares across accuracies

### Step 4: Validate Monotonicity
Check that performance improves from 60% → 70% → 80% → 90% → 100%

### Step 5: If Issues Found, Debug with Diagnostics
```bash
# Layer-by-layer investigation per SYNTHETIC_PREDICTION_TEST_PLAN.md
cd diagnostics/
jupyter lab diagnostic_17_paradox_analysis.ipynb
```

---

## Files in Diagnostics Folder

### Testing Infrastructure
- `diagnostic_100_algorithm_validation.py` - ⚡ NEW: 100% accuracy test
- `SYNTHETIC_PREDICTION_TEST_PLAN.md` - Layer-by-layer debugging strategy
- `test_all_strategies.py` - Quick strategy smoke tests
- `all_strategies_pct.py` - Strategy implementations

### Analysis Notebooks
- `diagnostic_17_paradox_analysis.ipynb` - Trade-by-trade investigation
- `diagnostic_16_optuna_with_params.ipynb` - Parameter optimization

### Support Files
- `cost_config_small_farmer.py` - Cost configurations
- `corrected_strategies.py` - Previous strategy fixes
- `fixed_strategies.py` - Alternate implementations

---

## Expected Results Summary

| Accuracy | Expected Net Earnings | vs Baseline | Status |
|----------|----------------------|-------------|---------|
| 100% (Perfect) | >$800k | +10%+ | ✓ Algorithms work |
| 90% | $755k-$775k | +4% to +7% | Target |
| 80% | $740k-$760k | +2% to +5% | Good |
| 70% | $730k-$745k | +0% to +3% | Break-even |
| 60% | $720k-$735k | -1% to +1% | Below threshold |
| Baseline | $727k | 0% | Reference |

---

## Key Insights

### Why 100% Accuracy Test is Critical

**Without 100% test:**
- Can't distinguish between bad algorithms and bad predictions
- Don't know if the system CAN work, even in principle
- Waste time tuning predictions when algorithms are broken

**With 100% test:**
- Proves algorithms work (or reveals bugs immediately)
- Provides upper bound on possible performance
- Separates algorithm issues from prediction quality issues

### Why Monotonicity Matters

If 90% beats baseline but 100% doesn't:
- Likely a data issue (100% predictions not actually 100%)
- Or threshold is too low (overfitting to noise)

If neither 90% nor 100% beat baseline:
- **Algorithm bug** - decision logic is wrong

---

## Troubleshooting

### Problem: 100% accuracy test fails
**Diagnosis:** Run `diagnostic_17_paradox_analysis.ipynb` to find bug
**Fix:** Correct algorithm logic in diagnostics folder, validate, then update main code

### Problem: 90% fails but 100% passes
**Diagnosis:** Algorithms work, but 90% accuracy isn't good enough
**Options:**
- Improve prediction accuracy (better models)
- Tune strategy parameters (lower thresholds)
- Accept that 80%+ accuracy needed for value-add

### Problem: No monotonicity (random pattern)
**Diagnosis:** Predictions not being used correctly in decisions
**Fix:** Check prediction passing (diagnostic 08), decision logic (diagnostic 11-14)

### Problem: All prediction strategies beat baselines
**Diagnosis:** ✓ SUCCESS! System is working
**Next:** Focus on improving prediction accuracy for production

---

**Document Owner:** Claude Code (Diagnostics)
**Last Updated:** 2025-11-24
**Purpose:** Guide testing across accuracy spectrum with focus on algorithm validation
