"""
Daily Trading Recommendations

Generates actionable trading recommendations using latest predictions.
Run this daily when new forecasts are available.

Usage:
    python daily_recommendations.py --commodity coffee --model sarimax_auto_weather_v1
    python daily_recommendations.py --commodity sugar --all-models
"""

import sys
import os
from databricks import sql
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from datetime import datetime
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import strategies
from commodity_prediction_analysis.trading_prediction_analysis_multi_model import (
    ImmediateSaleStrategy,
    EqualBatchStrategy,
    PriceThresholdStrategy,
    MovingAverageStrategy,
    ConsensusStrategy,
    ExpectedValueStrategy,
    RiskAdjustedStrategy,
    PriceThresholdPredictive,
    MovingAveragePredictive
)

# Import data access
from data_access.forecast_loader import (
    get_available_models,
    load_forecast_distributions,
    transform_to_prediction_matrices
)

# Load environment variables
load_dotenv()


def get_latest_prediction(commodity, model_version, connection):
    """
    Get the most recent prediction for operational use.

    Returns:
        tuple: (prediction_matrix, forecast_date, generation_timestamp)
    """
    # Query for latest forecast_start_date for this model
    cursor = connection.cursor()
    cursor.execute("""
        SELECT MAX(forecast_start_date) as latest_date
        FROM commodity.forecast.distributions
        WHERE commodity = %s
          AND model_version = %s
          AND is_actuals = FALSE
    """, (commodity.capitalize(), model_version))

    result = cursor.fetchone()
    if result is None or result[0] is None:
        raise ValueError(f"No predictions found for {commodity} - {model_version}")

    latest_date = result[0]

    # Load just this date's predictions
    df = load_forecast_distributions(
        commodity=commodity.capitalize(),
        model_version=model_version,
        connection=connection,
        start_date=latest_date,
        end_date=latest_date
    )

    if len(df) == 0:
        raise ValueError(f"No data for latest date {latest_date}")

    # Transform to prediction matrix
    matrices = transform_to_prediction_matrices(df)

    if len(matrices) == 0:
        raise ValueError(f"Could not transform predictions")

    # Get the single matrix
    forecast_date = list(matrices.keys())[0]
    prediction_matrix = matrices[forecast_date]

    # Get generation timestamp
    generation_ts = df['generation_timestamp'].max()

    cursor.close()

    return prediction_matrix, forecast_date, generation_ts


def get_current_state(commodity, connection):
    """
    Get current state for operational decision-making.

    In production, this would query your inventory management system.
    For now, uses placeholder values.

    Returns:
        dict with: inventory, days_since_harvest, current_price, price_history
    """
    # Get latest price
    cursor = connection.cursor()

    # TODO: Replace with actual price query from your system
    # For now, using placeholder
    cursor.execute("""
        SELECT date, price
        FROM commodity.prices.daily
        WHERE commodity = %s
        ORDER BY date DESC
        LIMIT 100
    """, (commodity.capitalize(),))

    rows = cursor.fetchall()
    if len(rows) == 0:
        # Fallback to mock data
        print("  ⚠️  No price data found, using mock data")
        price_history = pd.DataFrame({
            'date': pd.date_range(end=datetime.now(), periods=100, freq='D'),
            'price': 100 + np.cumsum(np.random.randn(100) * 0.5)
        })
        current_price = price_history['price'].iloc[-1]
    else:
        price_history = pd.DataFrame(rows, columns=['date', 'price'])
        price_history = price_history.sort_values('date')
        current_price = price_history['price'].iloc[-1]

    cursor.close()

    # TODO: Get actual inventory from your system
    # For now, placeholder
    inventory = 35.5  # tons
    days_since_harvest = 45

    return {
        'inventory': inventory,
        'days_since_harvest': days_since_harvest,
        'current_price': current_price,
        'price_history': price_history,
        'current_date': datetime.now()
    }


def initialize_strategies(commodity_config):
    """
    Initialize all trading strategies with commodity-specific parameters.

    Returns:
        list of (strategy_name, strategy_object, needs_predictions)
    """
    strategies = []

    # Baseline strategies (don't use predictions)
    strategies.append((
        'Immediate Sale',
        ImmediateSaleStrategy(min_batch_size=5.0, sale_frequency_days=7),
        False
    ))

    strategies.append((
        'Equal Batches',
        EqualBatchStrategy(batch_size=0.25, frequency_days=30),
        False
    ))

    strategies.append((
        'Price Threshold',
        PriceThresholdStrategy(
            threshold_pct=0.05,
            batch_fraction=0.25,
            max_days_without_sale=60
        ),
        False
    ))

    strategies.append((
        'Moving Average',
        MovingAverageStrategy(
            ma_period=30,
            batch_fraction=0.25,
            max_days_without_sale=60
        ),
        False
    ))

    # Prediction-based strategies
    strategies.append((
        'Consensus',
        ConsensusStrategy(
            consensus_threshold=0.70,
            min_return=0.03,
            evaluation_day=14
        ),
        True
    ))

    strategies.append((
        'Expected Value',
        ExpectedValueStrategy(
            storage_cost_pct_per_day=commodity_config['storage_cost_pct_per_day'],
            transaction_cost_pct=commodity_config['transaction_cost_pct'],
            min_ev_improvement=50,
            baseline_batch=0.15,
            baseline_frequency=10
        ),
        True
    ))

    strategies.append((
        'Risk-Adjusted',
        RiskAdjustedStrategy(
            min_return=0.05,
            max_uncertainty=0.08,
            consensus_threshold=0.65,
            evaluation_day=14
        ),
        True
    ))

    strategies.append((
        'Price Threshold Predictive',
        PriceThresholdPredictive(
            threshold_pct=0.05,
            batch_fraction=0.25,
            max_days_without_sale=60,
            storage_cost_pct_per_day=commodity_config['storage_cost_pct_per_day'],
            transaction_cost_pct=commodity_config['transaction_cost_pct']
        ),
        True
    ))

    strategies.append((
        'Moving Average Predictive',
        MovingAveragePredictive(
            ma_period=30,
            batch_fraction=0.25,
            max_days_without_sale=60,
            storage_cost_pct_per_day=commodity_config['storage_cost_pct_per_day'],
            transaction_cost_pct=commodity_config['transaction_cost_pct']
        ),
        True
    ))

    return strategies


def generate_recommendations(state, prediction_matrix, commodity_config):
    """
    Generate recommendations for all strategies.

    Returns:
        DataFrame with recommendations
    """
    strategies = initialize_strategies(commodity_config)

    # Prepare parameters for decide() method
    day = state['days_since_harvest']
    inventory = state['inventory']
    current_price = state['current_price']
    price_history = state['price_history']

    recommendations = []

    for strategy_name, strategy_obj, needs_predictions in strategies:
        try:
            # Set harvest start (strategies need this)
            strategy_obj.set_harvest_start(day=0)

            # Get decision
            if needs_predictions:
                decision = strategy_obj.decide(
                    day=day,
                    inventory=inventory,
                    current_price=current_price,
                    price_history=price_history,
                    predictions=prediction_matrix
                )
            else:
                decision = strategy_obj.decide(
                    day=day,
                    inventory=inventory,
                    current_price=current_price,
                    price_history=price_history,
                    predictions=None
                )

            recommendations.append({
                'Strategy': strategy_name,
                'Action': decision['action'],
                'Quantity (tons)': decision['amount'],
                'Reasoning': decision['reason'],
                'Uses Predictions': 'Yes' if needs_predictions else 'No'
            })

        except Exception as e:
            recommendations.append({
                'Strategy': strategy_name,
                'Action': 'ERROR',
                'Quantity (tons)': 0,
                'Reasoning': str(e),
                'Uses Predictions': 'Yes' if needs_predictions else 'No'
            })

    return pd.DataFrame(recommendations)


def main():
    parser = argparse.ArgumentParser(description='Generate daily trading recommendations')
    parser.add_argument('--commodity', required=True, choices=['coffee', 'sugar'],
                       help='Commodity to analyze')
    parser.add_argument('--model', help='Specific model to use (e.g., sarimax_auto_weather_v1)')
    parser.add_argument('--all-models', action='store_true',
                       help='Generate recommendations for all available models')

    args = parser.parse_args()

    if not args.model and not args.all_models:
        print("Error: Must specify either --model or --all-models")
        sys.exit(1)

    # Commodity configuration
    COMMODITY_CONFIGS = {
        'coffee': {
            'commodity': 'coffee',
            'harvest_volume': 50,
            'harvest_windows': [(5, 9)],
            'storage_cost_pct_per_day': 0.025,
            'transaction_cost_pct': 0.25
        },
        'sugar': {
            'commodity': 'sugar',
            'harvest_volume': 50,
            'harvest_windows': [(4, 9)],
            'storage_cost_pct_per_day': 0.020,
            'transaction_cost_pct': 0.25
        }
    }

    commodity_config = COMMODITY_CONFIGS[args.commodity]

    # Connect to Databricks
    print("=" * 80)
    print("DAILY TRADING RECOMMENDATIONS")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Commodity: {args.commodity.upper()}")
    print("=" * 80)
    print()

    print("Connecting to Databricks...")
    connection = sql.connect(
        server_hostname=os.getenv("DATABRICKS_HOST", "").replace("https://", ""),
        http_path=os.getenv("DATABRICKS_HTTP_PATH"),
        access_token=os.getenv("DATABRICKS_TOKEN")
    )
    print("✓ Connected\n")

    # Get current state
    print("Loading current state...")
    state = get_current_state(args.commodity, connection)
    print(f"✓ Current state loaded")
    print(f"  Inventory: {state['inventory']} tons")
    print(f"  Current Price: ${state['current_price']:.2f}")
    print(f"  Days Since Harvest: {state['days_since_harvest']}")
    print()

    # Determine which models to process
    if args.all_models:
        models = get_available_models(args.commodity.capitalize(), connection)
        print(f"Processing {len(models)} models...")
    else:
        models = [args.model]
        print(f"Processing model: {args.model}")

    print()

    # Generate recommendations for each model
    for model in models:
        print("=" * 80)
        print(f"MODEL: {model}")
        print("=" * 80)

        try:
            # Get latest prediction
            prediction, forecast_date, generation_ts = get_latest_prediction(
                args.commodity, model, connection
            )

            print(f"Latest Prediction:")
            print(f"  Forecast Date: {forecast_date}")
            print(f"  Generated: {generation_ts}")
            print(f"  Simulation Paths: {prediction.shape[0]}")
            print(f"  Forecast Horizon: {prediction.shape[1]} days")
            print()

            # Generate recommendations
            recommendations = generate_recommendations(state, prediction, commodity_config)

            # Display
            print("Recommendations:")
            print(recommendations.to_string(index=False))
            print()

            # Highlight key recommendations
            sell_recs = recommendations[recommendations['Action'] == 'SELL']
            if len(sell_recs) > 0:
                total_sell = sell_recs['Quantity (tons)'].sum()
                print(f"📊 Summary: {len(sell_recs)}/{len(recommendations)} strategies recommend SELL")
                print(f"   Total recommended: {total_sell:.1f} tons ({total_sell/state['inventory']*100:.1f}% of inventory)")
            else:
                print(f"📊 Summary: All strategies recommend HOLD")

            print()

        except Exception as e:
            print(f"❌ Error processing {model}: {e}")
            print()
            continue

    connection.close()

    print("=" * 80)
    print("RECOMMENDATIONS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
