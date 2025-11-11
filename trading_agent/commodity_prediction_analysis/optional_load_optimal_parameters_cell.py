# COMMAND ----------

# MAGIC %md
# MAGIC ## OPTIONAL: Load Optimal Parameters from Grid Search
# MAGIC
# MAGIC **NOTE:** This cell is OPTIONAL. Only run if you want to use parameters from grid search.
# MAGIC
# MAGIC To use:
# MAGIC 1. Run parameter_grid_search_notebook.py first
# MAGIC 2. Uncomment the code below
# MAGIC 3. Set USE_OPTIMAL_PARAMETERS = True
# MAGIC 4. Run this cell before strategy initialization

# COMMAND ----------

# Set to True to load optimal parameters from grid search
USE_OPTIMAL_PARAMETERS = False  # Change to True to use grid search results

# Path to optimal parameters file
OPTIMAL_PARAMETERS_PATH = '/dbfs/FileStore/optimal_parameters.json'

if USE_OPTIMAL_PARAMETERS:
    import json
    from pathlib import Path

    # Check if file exists
    if Path(OPTIMAL_PARAMETERS_PATH).exists():
        print("=" * 80)
        print("LOADING OPTIMAL PARAMETERS FROM GRID SEARCH")
        print("=" * 80)

        # Load optimal parameters
        with open(OPTIMAL_PARAMETERS_PATH, 'r') as f:
            optimal_config = json.load(f)

        print(f"\nLoaded optimal parameters:")
        print(f"  Generated: {optimal_config['generated_at']}")
        print(f"  Commodity: {optimal_config['commodity']}")
        print(f"  Model: {optimal_config['model']}")

        # Verify commodity matches
        if optimal_config['commodity'].lower() != CURRENT_COMMODITY.lower():
            print(f"\n⚠️  WARNING: Optimal parameters are for {optimal_config['commodity']}, "
                  f"but current commodity is {CURRENT_COMMODITY}")
            print("   Proceeding anyway, but results may not be optimal.")

        # Extract parameters
        opt_params = optimal_config['parameters']

        # Update BASELINE_PARAMS
        if 'equal_batch' in opt_params:
            BASELINE_PARAMS['equal_batch'] = opt_params['equal_batch']['params']
            print(f"\n✓ Updated equal_batch: {BASELINE_PARAMS['equal_batch']}")

        if 'price_threshold' in opt_params:
            BASELINE_PARAMS['price_threshold']['threshold_pct'] = \
                opt_params['price_threshold']['params']['threshold_pct']
            print(f"✓ Updated price_threshold.threshold_pct: "
                  f"{BASELINE_PARAMS['price_threshold']['threshold_pct']}")

        if 'moving_average' in opt_params:
            BASELINE_PARAMS['moving_average']['ma_period'] = \
                opt_params['moving_average']['params']['ma_period']
            print(f"✓ Updated moving_average.ma_period: "
                  f"{BASELINE_PARAMS['moving_average']['ma_period']}")

        # Update PREDICTION_PARAMS
        if 'consensus' in opt_params:
            PREDICTION_PARAMS['consensus'] = opt_params['consensus']['params']
            print(f"✓ Updated consensus: {PREDICTION_PARAMS['consensus']}")

        if 'expected_value' in opt_params:
            PREDICTION_PARAMS['expected_value'] = opt_params['expected_value']['params']
            print(f"✓ Updated expected_value: {PREDICTION_PARAMS['expected_value']}")

        if 'risk_adjusted' in opt_params:
            PREDICTION_PARAMS['risk_adjusted'] = opt_params['risk_adjusted']['params']
            print(f"✓ Updated risk_adjusted: {PREDICTION_PARAMS['risk_adjusted']}")

        # Store matched pair parameters for strategy initialization
        MATCHED_PAIR_PARAMS = {}

        if 'price_threshold' in opt_params:
            MATCHED_PAIR_PARAMS['price_threshold'] = {
                'threshold_pct': opt_params['price_threshold']['params']['threshold_pct'],
                'batch_fraction': opt_params['price_threshold']['params']['batch_fraction'],
                'max_days_without_sale': opt_params['price_threshold']['params']['max_days_without_sale']
            }
            print(f"\n✓ Price Threshold matched pair params: {MATCHED_PAIR_PARAMS['price_threshold']}")

        if 'moving_average' in opt_params:
            MATCHED_PAIR_PARAMS['moving_average'] = {
                'ma_period': opt_params['moving_average']['params']['ma_period'],
                'batch_fraction': opt_params['moving_average']['params']['batch_fraction'],
                'max_days_without_sale': opt_params['moving_average']['params']['max_days_without_sale']
            }
            print(f"✓ Moving Average matched pair params: {MATCHED_PAIR_PARAMS['moving_average']}")

        print("\n" + "=" * 80)
        print("OPTIMAL PARAMETERS LOADED SUCCESSFULLY")
        print("=" * 80)
        print("\nNOTE: You must re-run the strategy initialization cell below for changes to take effect")

    else:
        print(f"\n⚠️  Optimal parameters file not found: {OPTIMAL_PARAMETERS_PATH}")
        print("   Run parameter_grid_search_notebook.py first to generate optimal parameters")
        print("   Using default BASELINE_PARAMS and PREDICTION_PARAMS")
        USE_OPTIMAL_PARAMETERS = False
        MATCHED_PAIR_PARAMS = {}

else:
    print("Using default BASELINE_PARAMS and PREDICTION_PARAMS (from lines 66-98)")
    MATCHED_PAIR_PARAMS = {}

# COMMAND ----------

# MAGIC %md
# MAGIC ## OPTIONAL: Update Strategy Initialization to Use Optimal Parameters
# MAGIC
# MAGIC If USE_OPTIMAL_PARAMETERS = True, replace the strategy initialization cell (lines ~3462-3494)
# MAGIC with this updated version:

# COMMAND ----------

# EXAMPLE: Updated strategy initialization cell (ONLY USE IF USE_OPTIMAL_PARAMETERS = True)
#
# print("\nInitializing strategies...")
#
# # Use matched pair params if available, otherwise use defaults
# if USE_OPTIMAL_PARAMETERS and MATCHED_PAIR_PARAMS:
#     pt_params = MATCHED_PAIR_PARAMS.get('price_threshold', {
#         'threshold_pct': BASELINE_PARAMS['price_threshold']['threshold_pct'],
#         'batch_fraction': 0.25,
#         'max_days_without_sale': 60
#     })
#     ma_params = MATCHED_PAIR_PARAMS.get('moving_average', {
#         'ma_period': BASELINE_PARAMS['moving_average']['ma_period'],
#         'batch_fraction': 0.25,
#         'max_days_without_sale': 60
#     })
# else:
#     pt_params = {
#         'threshold_pct': BASELINE_PARAMS['price_threshold']['threshold_pct'],
#         'batch_fraction': 0.25,
#         'max_days_without_sale': 60
#     }
#     ma_params = {
#         'ma_period': BASELINE_PARAMS['moving_average']['ma_period'],
#         'batch_fraction': 0.25,
#         'max_days_without_sale': 60
#     }
#
# baselines = [
#     ImmediateSaleStrategy(),
#     EqualBatchStrategy(**BASELINE_PARAMS['equal_batch']),
#     PriceThresholdStrategy(**pt_params),
#     MovingAverageStrategy(**ma_params)
# ]
#
# prediction_strategies = [
#     ConsensusStrategy(**PREDICTION_PARAMS['consensus']),
#     ExpectedValueStrategy(
#         storage_cost_pct_per_day=commodity_config['storage_cost_pct_per_day'],
#         transaction_cost_pct=commodity_config['transaction_cost_pct'],
#         **PREDICTION_PARAMS['expected_value']
#     ),
#     RiskAdjustedStrategy(**PREDICTION_PARAMS['risk_adjusted']),
#     PriceThresholdPredictive(**pt_params),
#     MovingAveragePredictive(**ma_params)
# ]
#
# all_strategies = baselines + prediction_strategies
#
# print(f"✓ {len(baselines)} baseline strategies")
# print(f"✓ {len(prediction_strategies)} prediction-based strategies")
# if USE_OPTIMAL_PARAMETERS:
#     print(f"✓ Using optimal parameters from grid search")
# print(f"Total: {len(all_strategies)} strategies to test")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Integration Instructions
# MAGIC
# MAGIC To integrate optimal parameters into the main notebook:
# MAGIC
# MAGIC ### Option 1: Load Dynamically (Recommended for Testing)
# MAGIC 1. Insert the first cell above BEFORE the strategy initialization cell
# MAGIC 2. Set `USE_OPTIMAL_PARAMETERS = True`
# MAGIC 3. Run the cell
# MAGIC 4. Parameters will be loaded and applied automatically
# MAGIC
# MAGIC ### Option 2: Hard-Code (Recommended for Production)
# MAGIC 1. Run grid search and review optimal parameters
# MAGIC 2. Manually update `BASELINE_PARAMS` and `PREDICTION_PARAMS` (lines 66-98)
# MAGIC 3. Update strategy initialization (lines 3484-3492) with optimal values
# MAGIC 4. No need to load from file
# MAGIC
# MAGIC ### Validation
# MAGIC After loading optimal parameters:
# MAGIC 1. Re-run the strategy initialization cell
# MAGIC 2. Run full backtest
# MAGIC 3. Compare net earnings to baseline (pre-optimization)
# MAGIC 4. Check statistical significance
# MAGIC 5. If improvement is significant and validated, commit changes
