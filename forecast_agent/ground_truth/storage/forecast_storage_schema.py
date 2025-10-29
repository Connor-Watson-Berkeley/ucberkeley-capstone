"""Production forecast storage schema.

Stores forecasts and prediction intervals for trading agent consumption.
Designed for production deployment with weekly retraining.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import os


class ForecastStore:
    """
    Stores production forecasts with metadata and prediction intervals.

    Schema:
        - forecast_id: Unique identifier (model_name + inference_date)
        - model_id: Model version from ModelMetadataStore
        - model_name: Human-readable model name
        - commodity: Coffee, Sugar, etc.
        - inference_date: Date when forecast was generated
        - forecast_horizon: Number of days ahead
        - trained_on_days: Number of days used for training
        - training_end_date: Last date in training data
        - forecast_dates: List of forecast dates
        - forecast_values: Point forecasts
        - lower_bound_80: 80% prediction interval lower bound
        - upper_bound_80: 80% prediction interval upper bound
        - lower_bound_95: 95% prediction interval lower bound
        - upper_bound_95: 95% prediction interval upper bound
        - distribution_params: JSON of distribution parameters (mean, std, etc.)
        - status: active, archived, superseded
        - created_at: Timestamp
    """

    def __init__(self, storage_path: str = "production_forecasts.parquet"):
        """
        Initialize forecast store.

        Args:
            storage_path: Path to parquet file for storage
        """
        self.storage_path = storage_path

        if os.path.exists(storage_path):
            self.forecasts = pd.read_parquet(storage_path)
        else:
            self._create_empty_store()

    def _create_empty_store(self):
        """Create empty forecast store with schema."""
        self.forecasts = pd.DataFrame(columns=[
            'forecast_id',
            'model_id',
            'model_name',
            'commodity',
            'inference_date',
            'forecast_horizon',
            'trained_on_days',
            'training_end_date',
            'forecast_dates',
            'forecast_values',
            'lower_bound_80',
            'upper_bound_80',
            'lower_bound_95',
            'upper_bound_95',
            'distribution_params',
            'status',
            'created_at'
        ])

    def store_forecast(self,
                      model_id: str,
                      model_name: str,
                      commodity: str,
                      inference_date: datetime,
                      forecast_df: pd.DataFrame,
                      training_end_date: datetime,
                      trained_on_days: int,
                      prediction_intervals: Optional[Dict] = None,
                      distribution_params: Optional[Dict] = None) -> str:
        """
        Store a production forecast.

        Args:
            model_id: Model version ID from ModelMetadataStore
            model_name: Human-readable model name
            commodity: Commodity name
            inference_date: Date when forecast was generated
            forecast_df: DataFrame with columns ['date', 'forecast']
            training_end_date: Last date in training data
            trained_on_days: Number of days used for training
            prediction_intervals: Dict with 'lower_80', 'upper_80', 'lower_95', 'upper_95'
            distribution_params: Dict with distribution parameters (mean, std, skew, etc.)

        Returns:
            forecast_id: Unique identifier for this forecast
        """
        # Generate forecast ID
        forecast_id = f"{model_name.replace(' ', '_')}_{inference_date.strftime('%Y%m%d')}_{int(datetime.now().timestamp())}"

        # Extract forecast data
        forecast_dates = forecast_df['date'].tolist()
        forecast_values = forecast_df['forecast'].tolist()

        # Prediction intervals (default to None if not provided)
        lower_80 = prediction_intervals.get('lower_80', [None] * len(forecast_values)) if prediction_intervals else [None] * len(forecast_values)
        upper_80 = prediction_intervals.get('upper_80', [None] * len(forecast_values)) if prediction_intervals else [None] * len(forecast_values)
        lower_95 = prediction_intervals.get('lower_95', [None] * len(forecast_values)) if prediction_intervals else [None] * len(forecast_values)
        upper_95 = prediction_intervals.get('upper_95', [None] * len(forecast_values)) if prediction_intervals else [None] * len(forecast_values)

        # Build row
        row = {
            'forecast_id': forecast_id,
            'model_id': model_id,
            'model_name': model_name,
            'commodity': commodity,
            'inference_date': inference_date,
            'forecast_horizon': len(forecast_values),
            'trained_on_days': trained_on_days,
            'training_end_date': training_end_date,
            'forecast_dates': json.dumps([d.isoformat() if hasattr(d, 'isoformat') else str(d) for d in forecast_dates]),
            'forecast_values': json.dumps(forecast_values),
            'lower_bound_80': json.dumps(lower_80),
            'upper_bound_80': json.dumps(upper_80),
            'lower_bound_95': json.dumps(lower_95),
            'upper_bound_95': json.dumps(upper_95),
            'distribution_params': json.dumps(distribution_params) if distribution_params else None,
            'status': 'active',
            'created_at': datetime.now()
        }

        # Append to store
        self.forecasts = pd.concat([self.forecasts, pd.DataFrame([row])], ignore_index=True)

        # Save
        self.save()

        return forecast_id

    def get_active_forecast(self, commodity: str, model_name: str = None) -> Optional[Dict]:
        """
        Get most recent active forecast for a commodity.

        Args:
            commodity: Commodity name
            model_name: Optional - filter by model name

        Returns:
            Dict with forecast data or None
        """
        df = self.forecasts[
            (self.forecasts['commodity'] == commodity) &
            (self.forecasts['status'] == 'active')
        ].copy()

        if model_name:
            df = df[df['model_name'] == model_name]

        if len(df) == 0:
            return None

        # Get most recent
        df = df.sort_values('inference_date', ascending=False)
        latest = df.iloc[0]

        return self._row_to_dict(latest)

    def get_forecast_by_id(self, forecast_id: str) -> Optional[Dict]:
        """
        Get forecast by ID.

        Args:
            forecast_id: Forecast identifier

        Returns:
            Dict with forecast data or None
        """
        matches = self.forecasts[self.forecasts['forecast_id'] == forecast_id]

        if len(matches) == 0:
            return None

        return self._row_to_dict(matches.iloc[0])

    def _row_to_dict(self, row) -> Dict:
        """Convert forecast row to dictionary with parsed JSON."""
        return {
            'forecast_id': row['forecast_id'],
            'model_id': row['model_id'],
            'model_name': row['model_name'],
            'commodity': row['commodity'],
            'inference_date': row['inference_date'],
            'forecast_horizon': int(row['forecast_horizon']),
            'trained_on_days': int(row['trained_on_days']),
            'training_end_date': row['training_end_date'],
            'forecast_dates': json.loads(row['forecast_dates']),
            'forecast_values': json.loads(row['forecast_values']),
            'lower_bound_80': json.loads(row['lower_bound_80']),
            'upper_bound_80': json.loads(row['upper_bound_80']),
            'lower_bound_95': json.loads(row['lower_bound_95']),
            'upper_bound_95': json.loads(row['upper_bound_95']),
            'distribution_params': json.loads(row['distribution_params']) if pd.notna(row['distribution_params']) else None,
            'status': row['status'],
            'created_at': row['created_at']
        }

    def archive_old_forecasts(self, commodity: str, model_name: str, keep_latest_n: int = 5):
        """
        Archive old forecasts, keeping only the N most recent.

        Args:
            commodity: Commodity name
            model_name: Model name
            keep_latest_n: Number of recent forecasts to keep active
        """
        df = self.forecasts[
            (self.forecasts['commodity'] == commodity) &
            (self.forecasts['model_name'] == model_name) &
            (self.forecasts['status'] == 'active')
        ].copy()

        if len(df) <= keep_latest_n:
            return

        # Sort by inference date
        df = df.sort_values('inference_date', ascending=False)

        # Archive older ones
        to_archive = df.iloc[keep_latest_n:]['forecast_id'].tolist()

        self.forecasts.loc[
            self.forecasts['forecast_id'].isin(to_archive),
            'status'
        ] = 'archived'

        self.save()

    def export_for_trading_agent(self, commodity: str, output_path: str = "trading_agent_forecast.json"):
        """
        Export latest forecast in format for trading agent.

        Args:
            commodity: Commodity name
            output_path: Path to JSON file
        """
        forecast = self.get_active_forecast(commodity)

        if forecast is None:
            print(f"✗ No active forecast found for {commodity}")
            return None

        # Format for trading agent
        trading_forecast = {
            'commodity': forecast['commodity'],
            'model': forecast['model_name'],
            'inference_date': str(forecast['inference_date']),
            'horizon_days': forecast['forecast_horizon'],
            'forecasts': []
        }

        # Build forecast entries
        for i in range(len(forecast['forecast_dates'])):
            entry = {
                'date': forecast['forecast_dates'][i],
                'forecast': forecast['forecast_values'][i],
                'lower_80': forecast['lower_bound_80'][i],
                'upper_80': forecast['upper_bound_80'][i],
                'lower_95': forecast['lower_bound_95'][i],
                'upper_95': forecast['upper_bound_95'][i]
            }
            trading_forecast['forecasts'].append(entry)

        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(trading_forecast, f, indent=2)

        print(f"✓ Exported forecast to {output_path}")
        print(f"  Model: {forecast['model_name']}")
        print(f"  Inference Date: {forecast['inference_date']}")
        print(f"  Horizon: {forecast['forecast_horizon']} days")

        return trading_forecast

    def save(self):
        """Save forecast store to disk."""
        self.forecasts.to_parquet(self.storage_path, index=False)


# Example production usage
EXAMPLE_PRODUCTION_USAGE = """
# PRODUCTION DEPLOYMENT WORKFLOW
# ================================

from ground_truth.storage.forecast_storage_schema import ForecastStore
from ground_truth.storage.model_metadata_schema import ModelMetadataStore
from ground_truth.models import xgboost_model
import pandas as pd
from datetime import datetime

# 1. Load full historical data
df = pd.read_parquet("unified_data_with_gdelt.parquet")
df_coffee = df[df['commodity'] == 'Coffee'].set_index('date').sort_index()

# 2. Get best model from metadata store
model_store = ModelMetadataStore("model_registry.parquet")
best_model = model_store.get_best_model(
    commodity='Coffee',
    metric='mae'  # or 'directional_accuracy_from_day0' for trading
)

# 3. Train on FULL history (no train/test split in production)
print(f"Training {best_model['model_name']} on full history...")
inference_date = datetime.now()

result = xgboost_model.xgboost_forecast_with_metadata(
    df_pandas=df_coffee,
    commodity='Coffee',
    target='close',
    horizon=14,
    exog_features=['temp_c', 'humidity_pct', 'precipitation_mm'],
    lags=[1, 2, 3, 7, 14, 21, 30],
    windows=[7, 14, 30, 60]
)

# 4. Store forecast in production store
forecast_store = ForecastStore("production_forecasts.parquet")

forecast_id = forecast_store.store_forecast(
    model_id=best_model['model_id'],
    model_name=best_model['model_name'],
    commodity='Coffee',
    inference_date=inference_date,
    forecast_df=result['forecast_df'],
    training_end_date=df_coffee.index[-1],
    trained_on_days=len(df_coffee),
    prediction_intervals=None,  # TODO: Add prediction intervals
    distribution_params={
        'residual_std': 2.5,  # From walk-forward evaluation
        'mae': best_model['mae']
    }
)

print(f"✓ Forecast stored: {forecast_id}")

# 5. Export for trading agent
forecast_store.export_for_trading_agent('Coffee', 'trading_agent_forecast.json')

# 6. Archive old forecasts (keep latest 5)
forecast_store.archive_old_forecasts('Coffee', best_model['model_name'], keep_latest_n=5)

# WEEKLY RETRAINING SCRIPT
# =========================
# Run this script every Monday morning:

# 1. Load latest data (including last week's actuals)
# 2. Retrain model on full history
# 3. Generate new 14-day forecast
# 4. Store in ForecastStore
# 5. Export for trading agent
# 6. Archive old forecasts
# 7. Monitor: Compare last week's forecast vs actuals
"""
