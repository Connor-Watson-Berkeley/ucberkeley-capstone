#!/usr/bin/env python3
"""
Simple test script to verify production backtest functionality.

This tests:
1. Loading data and predictions
2. Running production BacktestEngine
3. Basic strategy execution (ImmediateSaleStrategy)
"""
import sys
from pathlib import Path

# Add parent directory to path
try:
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir.parent))
except NameError:
    # __file__ not defined in Databricks jobs
    sys.path.insert(0, '/Workspace/Repos/Project_Git/ucberkeley-capstone/trading_agent/commodity_prediction_analysis')

import pandas as pd
import pickle
from production.config import COMMODITY_CONFIGS, VOLUME_PATH
from production.core.backtest_engine import BacktestEngine
from production.strategies.baseline import ImmediateSaleStrategy

def test_production_backtest(commodity='coffee'):
    """Test production backtest with ImmediateSaleStrategy."""

    print(f"\n{'='*80}")
    print(f"TESTING PRODUCTION BACKTEST - {commodity.upper()}")
    print(f"{'='*80}\n")

    # Get commodity config
    config = COMMODITY_CONFIGS.get(commodity)
    if not config:
        print(f"❌ ERROR: No config found for commodity '{commodity}'")
        return False

    print(f"✓ Loaded config for {commodity}")
    print(f"  - Harvest volume: {config['harvest_volume']:,} bags")
    print(f"  - Harvest windows: {len(config['harvest_windows'])}")
    print(f"  - Storage cost: {config['storage_cost_pct_per_day']*100:.4f}%/day")
    print(f"  - Transaction cost: {config['transaction_cost_pct']*100:.2f}%")

    # Load prices
    try:
        prices_path = f"{VOLUME_PATH}/{commodity}_prices.pkl"
        with open(prices_path, 'rb') as f:
            prices = pickle.load(f)
        print(f"\n✓ Loaded prices: {len(prices)} rows")
        print(f"  - Date range: {prices.index.min()} to {prices.index.max()}")
        print(f"  - Price range: ${prices['price'].min():.2f} - ${prices['price'].max():.2f}")
    except Exception as e:
        print(f"❌ ERROR loading prices: {e}")
        return False

    # Load predictions
    try:
        predictions_path = f"{VOLUME_PATH}/{commodity}_arima_v1_predictions.pkl"
        with open(predictions_path, 'rb') as f:
            predictions = pickle.load(f)
        print(f"\n✓ Loaded predictions")
        print(f"  - Keys: {list(predictions.keys())}")
    except Exception as e:
        print(f"❌ ERROR loading predictions: {e}")
        return False

    # Initialize BacktestEngine
    try:
        engine = BacktestEngine(prices, predictions, config)
        print(f"\n✓ Initialized BacktestEngine")
    except Exception as e:
        print(f"❌ ERROR initializing BacktestEngine: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Initialize ImmediateSaleStrategy (simplest baseline)
    try:
        strategy = ImmediateSaleStrategy(
            min_batch_size=config.get('min_batch_size', 50),
            sale_frequency_days=config.get('sale_frequency_days', 7),
            storage_cost_pct_per_day=config['storage_cost_pct_per_day'],
            transaction_cost_pct=config['transaction_cost_pct']
        )
        print(f"\n✓ Initialized ImmediateSaleStrategy")
        print(f"  - Min batch size: {strategy.min_batch_size} bags")
        print(f"  - Sale frequency: {strategy.sale_frequency_days} days")
    except Exception as e:
        print(f"❌ ERROR initializing strategy: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Run backtest
    try:
        print(f"\n{'='*80}")
        print("RUNNING BACKTEST...")
        print(f"{'='*80}\n")

        results = engine.run_backtest(strategy)

        print(f"\n{'='*80}")
        print("BACKTEST RESULTS")
        print(f"{'='*80}\n")

        print(f"✓ Backtest completed successfully!")
        print(f"\n  Earnings: ${results['total_earnings']:,.2f}")
        print(f"  Transactions: {len(results['transaction_history'])}")
        print(f"  Final inventory: {results['final_inventory']:,.0f} bags")

        if results['transaction_history']:
            print(f"\n  First transaction:")
            first_tx = results['transaction_history'][0]
            print(f"    Date: {first_tx['date']}")
            print(f"    Volume: {first_tx['volume']:,.0f} bags")
            print(f"    Price: ${first_tx['price']:.2f}")
            print(f"    Revenue: ${first_tx['revenue']:,.2f}")

            print(f"\n  Last transaction:")
            last_tx = results['transaction_history'][-1]
            print(f"    Date: {last_tx['date']}")
            print(f"    Volume: {last_tx['volume']:,.0f} bags")
            print(f"    Price: ${last_tx['price']:.2f}")
            print(f"    Revenue: ${last_tx['revenue']:,.2f}")

        return True

    except Exception as e:
        print(f"❌ ERROR running backtest: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    # Test with coffee
    success = test_production_backtest('coffee')

    if success:
        print(f"\n{'='*80}")
        print("✅ PRODUCTION BACKTEST TEST PASSED")
        print(f"{'='*80}\n")
        sys.exit(0)
    else:
        print(f"\n{'='*80}")
        print("❌ PRODUCTION BACKTEST TEST FAILED")
        print(f"{'='*80}\n")
        sys.exit(1)
