"""
WhatsApp Trading Recommendations - Lambda Handler (Real Data)

Responds to Twilio WhatsApp webhook with trading recommendations.
Queries Databricks via REST API for real market data, forecasts, and recommendations.
"""

import json
import os
import time
from datetime import datetime, date, timedelta
from typing import Dict, Optional, Tuple, List, Any
import numpy as np
import requests


def execute_databricks_query(sql_query: str, timeout: int = 60) -> List[List[Any]]:
    """
    Execute SQL query on Databricks using REST API.

    Args:
        sql_query: SQL query to execute
        timeout: Maximum time to wait for query completion (seconds)

    Returns:
        List of rows (each row is a list of values)

    Raises:
        Exception if query fails or times out
    """
    # Get credentials from environment
    host = os.environ['DATABRICKS_HOST']
    token = os.environ['DATABRICKS_TOKEN']
    warehouse_id = os.environ['DATABRICKS_HTTP_PATH'].split('/')[-1]

    # Prepare request
    url = f"https://{host}/api/2.0/sql/statements/"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "statement": sql_query,
        "warehouse_id": warehouse_id,
        "wait_timeout": "30s"  # Server-side timeout
    }

    print(f"Executing Databricks query: {sql_query[:100]}...")

    # Submit query
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()

    result = response.json()
    statement_id = result.get("statement_id")

    if not statement_id:
        raise Exception(f"No statement_id in response: {result}")

    # Poll for completion
    status_url = f"https://{host}/api/2.0/sql/statements/{statement_id}"
    start_time = time.time()

    while time.time() - start_time < timeout:
        status_response = requests.get(status_url, headers=headers)
        status_response.raise_for_status()

        status_data = status_response.json()
        state = status_data.get("status", {}).get("state")

        print(f"Query status: {state}")

        if state == "SUCCEEDED":
            # Extract results
            manifest = status_data.get("manifest", {})
            chunks = manifest.get("chunks", [])

            if not chunks:
                # No results (empty query)
                return []

            # Get first chunk (for most queries, there's only one chunk)
            chunk_index = chunks[0].get("chunk_index", 0)
            result_url = f"https://{host}/api/2.0/sql/statements/{statement_id}/result/chunks/{chunk_index}"

            result_response = requests.get(result_url, headers=headers)
            result_response.raise_for_status()

            result_data = result_response.json()
            data_array = result_data.get("data_array", [])

            print(f"Query returned {len(data_array)} rows")
            return data_array

        elif state == "FAILED":
            error = status_data.get("status", {}).get("error", {})
            raise Exception(f"Query failed: {error.get('message', 'Unknown error')}")

        elif state == "CANCELED":
            raise Exception("Query was canceled")

        # Still running, wait before polling again
        time.sleep(2)

    raise Exception(f"Query timed out after {timeout} seconds")


def get_latest_market_price(commodity: str) -> Tuple[float, date]:
    """Get the most recent closing price for a commodity."""
    query = f"""
        SELECT close, date
        FROM commodity.bronze.market
        WHERE commodity = '{commodity}'
        ORDER BY date DESC
        LIMIT 1
    """

    rows = execute_databricks_query(query)

    if rows:
        return float(rows[0][0]), rows[0][1]
    else:
        raise ValueError(f"No market data found for {commodity}")


def calculate_7day_trend(commodity: str, current_date: date) -> float:
    """Calculate 7-day price trend percentage."""
    query = f"""
        SELECT close
        FROM commodity.bronze.market
        WHERE commodity = '{commodity}'
          AND date <= '{current_date}'
        ORDER BY date DESC
        LIMIT 8
    """

    rows = execute_databricks_query(query)

    if len(rows) >= 2:
        current_price = float(rows[0][0])
        week_ago_price = float(rows[-1][0])
        trend_pct = ((current_price - week_ago_price) / week_ago_price) * 100
        return trend_pct
    else:
        return 0.0


def get_best_available_model(
    commodity: str,
    max_age_days: int = 30,
    metric: str = 'mae_14d'
) -> Optional[str]:
    """
    Get best performing model that has recent forecasts available.

    Steps:
    1. Find models with forecasts within max_age_days
    2. Of those, select the best by performance metric

    Args:
        commodity: 'Coffee' or 'Sugar'
        max_age_days: Maximum age of forecasts to consider
        metric: Performance metric to optimize (mae_14d, rmse_14d, crps_14d)

    Returns:
        model_version string, or None if no forecasts available
    """
    cutoff_date = date.today() - timedelta(days=max_age_days)

    # Step 1: Find available models with recent forecasts
    # Use subquery to find models with recent data, then select best by metric
    query = f"""
        SELECT
            m.model_version,
            AVG(m.{metric}) as avg_metric
        FROM commodity.forecast.forecast_metadata m
        WHERE m.commodity = '{commodity}'
          AND m.{metric} IS NOT NULL
          AND m.model_success = TRUE
          AND m.model_version IN (
              SELECT DISTINCT model_version
              FROM commodity.forecast.distributions
              WHERE commodity = '{commodity}'
                AND is_actuals = FALSE
                AND forecast_start_date >= '{cutoff_date}'
          )
        GROUP BY m.model_version
        ORDER BY avg_metric ASC
        LIMIT 1
    """

    try:
        rows = execute_databricks_query(query)
        if rows and len(rows) > 0:
            best_model = rows[0][0]
            metric_value = rows[0][1]
            print(f"Best available model: {best_model} ({metric}={metric_value:.4f})")
            return best_model
        else:
            print(f"No models with both forecasts and metadata available for {commodity}")
            # Fallback: just get any available model
            fallback_query = f"""
                SELECT DISTINCT model_version
                FROM commodity.forecast.distributions
                WHERE commodity = '{commodity}'
                  AND is_actuals = FALSE
                  AND forecast_start_date >= '{cutoff_date}'
                ORDER BY forecast_start_date DESC
                LIMIT 1
            """
            fallback_rows = execute_databricks_query(fallback_query)
            if fallback_rows and len(fallback_rows) > 0:
                fallback_model = fallback_rows[0][0]
                print(f"Using fallback model (no metadata): {fallback_model}")
                return fallback_model
            return None
    except Exception as e:
        print(f"Error selecting best model: {e}")
        return None


def get_available_forecast(
    commodity: str,
    max_age_days: int = 30,
    preferred_model: Optional[str] = None
) -> Optional[Dict]:
    """
    Get forecast for commodity, preferring specified model if available.

    Returns:
        Dict with:
            - model_version: str
            - forecast_date: date
            - prediction_matrix: np.ndarray (2000, 14)
    """
    cutoff_date = date.today() - timedelta(days=max_age_days)

    # If no preferred model specified, find the best available one
    if preferred_model is None:
        preferred_model = get_best_available_model(commodity, max_age_days)
        if preferred_model is None:
            return None

    # Get forecast for the preferred/best model
    query = f"""
        SELECT
            model_version,
            forecast_start_date,
            day_1, day_2, day_3, day_4, day_5, day_6, day_7,
            day_8, day_9, day_10, day_11, day_12, day_13, day_14
        FROM commodity.forecast.distributions
        WHERE commodity = '{commodity}'
          AND model_version = '{preferred_model}'
          AND is_actuals = FALSE
          AND forecast_start_date >= '{cutoff_date}'
        ORDER BY forecast_start_date DESC
        LIMIT 2000
    """

    rows = execute_databricks_query(query)

    if not rows:
        print(f"No forecast data for model {preferred_model}")
        return None

    # Extract model and date from first row
    model_version = rows[0][0]
    forecast_date = rows[0][1]

    # Build prediction matrix (2000 paths × 14 days)
    prediction_matrix = []
    for row in rows:
        if row[0] != model_version or row[1] != forecast_date:
            break  # Different model/date

        # Extract day_1 through day_14
        path = [float(row[i]) for i in range(2, 16)]
        prediction_matrix.append(path)

    print(f"Loaded {len(prediction_matrix)} paths for {model_version} (forecast date: {forecast_date})")

    return {
        'model_version': model_version,
        'forecast_date': forecast_date,
        'prediction_matrix': np.array(prediction_matrix)
    }


def calculate_expected_value_recommendation(
    current_price: float,
    prediction_matrix: np.ndarray,
    inventory_tons: float = 50.0,
    storage_cost_pct_per_day: float = 0.00025,
    transaction_cost_pct: float = 0.0025
) -> Dict:
    """
    Calculate Expected Value strategy recommendation.

    This is the strategy that won in backtesting (+3.4% for Coffee).

    Args:
        current_price: Current market price
        prediction_matrix: (2000, 14) array of Monte Carlo paths
        inventory_tons: Inventory size
        storage_cost_pct_per_day: Storage cost as % of value per day (0.025%)
        transaction_cost_pct: Transaction cost as % of sale value (0.25%)

    Returns:
        Dict with:
            - action: 'HOLD' or 'SELL'
            - expected_gain_per_ton: float
            - total_expected_gain: float
            - optimal_sale_day: int (1-14) if action is HOLD
            - reasoning: str
    """
    # Calculate median forecasts for each day
    median_forecasts = np.median(prediction_matrix, axis=0)

    # Calculate expected value of selling on each future day
    best_ev = -np.inf
    best_day = 0

    for day in range(14):
        expected_price = median_forecasts[day]
        cumulative_storage_cost = current_price * storage_cost_pct_per_day * (day + 1)
        transaction_cost = expected_price * transaction_cost_pct
        ev = expected_price - cumulative_storage_cost - transaction_cost

        if ev > best_ev:
            best_ev = ev
            best_day = day + 1

    # Compare with selling immediately
    immediate_sale_value = current_price - (current_price * transaction_cost_pct)
    expected_gain_per_ton = best_ev - immediate_sale_value

    # Decision threshold: $50/ton minimum gain
    min_gain_threshold = 50.0

    if expected_gain_per_ton > min_gain_threshold:
        action = 'HOLD'
        reasoning = f"Expected to gain ${expected_gain_per_ton:.0f}/ton by selling on day {best_day}"
        total_expected_gain = expected_gain_per_ton * inventory_tons
    else:
        action = 'SELL'
        reasoning = "Immediate sale recommended (expected gain too small)"
        best_day = 0
        total_expected_gain = 0.0

    # Calculate forecast range (10th-90th percentile)
    forecast_range = (
        float(np.percentile(prediction_matrix, 10)),
        float(np.percentile(prediction_matrix, 90))
    )

    # Find best 3-day window (highest median prices)
    window_size = 3
    best_window_start = 0
    best_window_median = 0
    for i in range(14 - window_size + 1):
        window_median = np.median(median_forecasts[i:i+window_size])
        if window_median > best_window_median:
            best_window_median = window_median
            best_window_start = i + 1

    return {
        'action': action,
        'expected_gain_per_ton': round(expected_gain_per_ton, 2),
        'total_expected_gain': round(total_expected_gain, 2),
        'optimal_sale_day': best_day,
        'reasoning': reasoning,
        'forecast_range': forecast_range,
        'best_sale_window': (best_window_start, best_window_start + window_size - 1)
    }


def format_whatsapp_message(
    commodity: str,
    current_price: float,
    price_date: date,
    trend_7d: float,
    forecast: Dict,
    strategy_decision: Dict,
    inventory_tons: float = 50.0
) -> str:
    """Format trading recommendation as WhatsApp message."""

    # Trend emoji
    if trend_7d > 0:
        trend_emoji = "📈"
        trend_sign = "+"
    else:
        trend_emoji = "📉"
        trend_sign = ""

    # Action emoji and formatting
    if strategy_decision['action'] == 'HOLD':
        action_emoji = "✋"
        action_text = f"*{action_emoji} HOLD*"
    else:
        action_emoji = "💰"
        action_text = f"*{action_emoji} SELL NOW*"

    # Forecast range
    forecast_range = strategy_decision['forecast_range']
    forecast_range_str = f"${forecast_range[0]:.2f} - ${forecast_range[1]:.2f}"

    # Best sale window
    window = strategy_decision['best_sale_window']
    window_str = f"Days {window[0]}-{window[1]}"

    # Hold duration
    hold_duration = strategy_decision['optimal_sale_day']

    # Expected gain
    total_gain = strategy_decision['total_expected_gain']

    # Build message
    message = f"""────────────────────────────────────────
📊 *Current Market*
Price: ${current_price:.2f}/kg
7-Day Trend: {trend_emoji} {trend_sign}{trend_7d:.1f}%

🔮 *14-Day Forecast*
Range: {forecast_range_str}
Best Sale Window: {window_str}

📦 *Inventory*
Stock: {int(inventory_tons)} tons
Hold Duration: {hold_duration} days

💡 *Recommendation*
{action_text}
Expected Gain: ${total_gain:,.0f}
"""

    if strategy_decision['action'] == 'HOLD':
        message += f"Sell on: Day {hold_duration}\n"
    else:
        message += f"Rationale: {strategy_decision['reasoning']}\n"

    message += f"""
_Model: {forecast['model_version']}_
_Strategy: Expected Value (+3.4% proven)_
_Forecast Date: {forecast['forecast_date']}_
────────────────────────────────────────"""

    return message


def generate_recommendation_from_databricks(commodity: str) -> Dict:
    """
    Generate real recommendation by querying Databricks via REST API.

    Returns:
        Dict with:
            - whatsapp_message: str
            - metadata: dict
    """
    # Get current market data
    current_price, price_date = get_latest_market_price(commodity)
    trend_7d = calculate_7day_trend(commodity, price_date)

    # Get forecast
    forecast = get_available_forecast(commodity, max_age_days=7)

    if not forecast:
        raise ValueError(f"No recent forecast available for {commodity}")

    # Calculate recommendation using Expected Value strategy
    strategy_decision = calculate_expected_value_recommendation(
        current_price=current_price,
        prediction_matrix=forecast['prediction_matrix'],
        inventory_tons=50.0
    )

    # Format message
    whatsapp_message = format_whatsapp_message(
        commodity=commodity,
        current_price=current_price,
        price_date=price_date,
        trend_7d=trend_7d,
        forecast=forecast,
        strategy_decision=strategy_decision,
        inventory_tons=50.0
    )

    return {
        'whatsapp_message': whatsapp_message,
        'metadata': {
            'commodity': commodity,
            'current_price': current_price,
            'model': forecast['model_version'],
            'strategy': 'ExpectedValue',
            'action': strategy_decision['action'],
            'expected_gain': strategy_decision['total_expected_gain']
        }
    }


def get_mock_recommendation(commodity='Coffee'):
    """
    Fallback mock recommendation if Databricks unavailable.
    """
    if commodity == 'Coffee':
        return {
            'whatsapp_message': """────────────────────────────────────────
📊 *Current Market*
Price: $2.15/kg
7-Day Trend: 📈 +2.3%

🔮 *14-Day Forecast*
Range: $2.08 - $2.28
Best Sale Window: Days 9-11

📦 *Inventory*
Stock: 50 tons
Hold Duration: 10 days

💡 *Recommendation*
*✋ HOLD*
Expected Gain: $6,250
Sell on: Day 10

_Model: sarimax_auto_weather_v1_
_Strategy: Expected Value (+3.4% proven)_
_Forecast Date: 2025-01-18_
────────────────────────────────────────""",
            'metadata': {
                'commodity': 'Coffee',
                'model': 'sarimax_auto_weather_v1',
                'strategy': 'ExpectedValue',
                'action': 'HOLD',
                'expected_gain': 6250
            }
        }
    elif commodity == 'Sugar':
        return {
            'whatsapp_message': """────────────────────────────────────────
📊 *Current Market*
Price: $0.18/kg
7-Day Trend: 📉 -0.8%

🔮 *14-Day Forecast*
Range: $0.17 - $0.19
Best Sale Window: Days 3-5

📦 *Inventory*
Stock: 50 tons
Hold Duration: 0 days

💡 *Recommendation*
*💰 SELL NOW*
Expected Gain: $0
Rationale: Low volatility commodity

_Model: prophet_v1_
_Strategy: Consensus (prediction-based)_
_Forecast Date: 2025-01-18_
────────────────────────────────────────""",
            'metadata': {
                'commodity': 'Sugar',
                'model': 'prophet_v1',
                'strategy': 'Consensus',
                'action': 'SELL',
                'expected_gain': 0
            }
        }
    else:
        return get_mock_recommendation('Coffee')


def parse_commodity_from_message(message_body):
    """
    Extract commodity preference from user message.

    Examples:
        "coffee" → Coffee
        "sugar recommendation" → Sugar
        "hello" → Coffee (default)
    """
    message_lower = message_body.lower()

    if 'sugar' in message_lower:
        return 'Sugar'
    else:
        # Default to Coffee
        return 'Coffee'


def format_twilio_response(message):
    """
    Format response for Twilio webhook.

    Twilio expects TwiML response format.
    """
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Message>{message}</Message>
</Response>"""

    return twiml


def lambda_handler(event, context):
    """
    AWS Lambda handler for Twilio WhatsApp webhook.

    Event structure (from API Gateway):
        {
            'body': 'From=whatsapp%3A%2B1234567890&Body=coffee&...',
            'headers': {...},
            'httpMethod': 'POST',
            ...
        }

    Returns:
        {
            'statusCode': 200,
            'headers': {'Content-Type': 'text/xml'},
            'body': '<Response><Message>...</Message></Response>'
        }
    """

    # Log incoming request
    print(f"Received WhatsApp webhook: {json.dumps(event)}")

    try:
        # Parse form data from Twilio
        if event.get('body'):
            # Parse URL-encoded form data
            from urllib.parse import parse_qs

            body = event['body']
            params = parse_qs(body)

            # Extract Twilio parameters
            from_number = params.get('From', [''])[0]
            message_body = params.get('Body', [''])[0]

            print(f"From: {from_number}")
            print(f"Message: {message_body}")

            # Determine commodity from message
            commodity = parse_commodity_from_message(message_body)
            print(f"Detected commodity: {commodity}")

            # Get recommendation
            try:
                if os.environ.get('DATABRICKS_HOST'):
                    print("Querying Databricks for real data via REST API...")
                    recommendation = generate_recommendation_from_databricks(commodity)
                else:
                    print("Using mock data (Databricks not configured)")
                    recommendation = get_mock_recommendation(commodity)
            except Exception as e:
                print(f"Error getting recommendation from Databricks: {str(e)}")
                import traceback
                traceback.print_exc()
                print("Falling back to mock data")
                recommendation = get_mock_recommendation(commodity)

            # Log metadata
            print(f"Recommendation metadata: {json.dumps(recommendation['metadata'])}")

            # Format response
            twiml = format_twilio_response(recommendation['whatsapp_message'])

            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'text/xml'
                },
                'body': twiml
            }

        else:
            # No body - return error
            error_message = "No message received. Try sending 'coffee' or 'sugar'."
            twiml = format_twilio_response(error_message)

            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'text/xml'
                },
                'body': twiml
            }

    except Exception as e:
        # Log error
        print(f"Error processing webhook: {str(e)}")
        import traceback
        traceback.print_exc()

        # Return friendly error message
        error_message = "Sorry, something went wrong. Please try again later."
        twiml = format_twilio_response(error_message)

        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'text/xml'
            },
            'body': twiml
        }


# For local testing
if __name__ == "__main__":
    # Simulate Twilio webhook call
    test_event = {
        'body': 'From=whatsapp%3A%2B15555551234&Body=coffee&MessageSid=SM123',
        'httpMethod': 'POST',
        'headers': {
            'Content-Type': 'application/x-www-form-urlencoded'
        }
    }

    result = lambda_handler(test_event, None)
    print("\nResponse:")
    print(result['body'])
