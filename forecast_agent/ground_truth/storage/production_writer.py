"""Production forecast writer aligned with commodity.silver schema.

Writes forecasts to three production tables:
- commodity.silver.point_forecasts
- commodity.silver.distributions
- commodity.silver.forecast_actuals
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os


class ProductionForecastWriter:
    """
    Writes forecasts to production tables following DATA_CONTRACTS.md schema.

    Tables:
        - point_forecasts: One row per (forecast_date, model_version, day_ahead)
        - distributions: One row per (forecast_start_date, model_version, path_id)
        - forecast_actuals: One row per (forecast_date, commodity)
    """

    def __init__(self, output_dir: str = "production_forecasts"):
        """
        Initialize production writer.

        Args:
            output_dir: Directory for parquet files (simulating DB tables)
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.point_forecasts_path = os.path.join(output_dir, "point_forecasts.parquet")
        self.distributions_path = os.path.join(output_dir, "distributions.parquet")
        self.forecast_actuals_path = os.path.join(output_dir, "forecast_actuals.parquet")

        # Load existing tables or create empty
        self._load_or_create_tables()

    def _load_or_create_tables(self):
        """Load existing tables or create empty DataFrames."""
        # Point forecasts table
        if os.path.exists(self.point_forecasts_path):
            self.point_forecasts = pd.read_parquet(self.point_forecasts_path)
        else:
            self.point_forecasts = pd.DataFrame(columns=[
                'forecast_date', 'data_cutoff_date', 'generation_timestamp',
                'day_ahead', 'forecast_mean', 'forecast_std',
                'lower_95', 'upper_95', 'model_version', 'commodity', 'model_success',
                'actual_close', 'has_data_leakage'  # New columns
            ])

        # Distributions table
        if os.path.exists(self.distributions_path):
            self.distributions = pd.read_parquet(self.distributions_path)
        else:
            day_cols = [f'day_{i}' for i in range(1, 15)]
            self.distributions = pd.DataFrame(columns=[
                'path_id', 'forecast_start_date', 'data_cutoff_date',
                'generation_timestamp', 'model_version', 'commodity',
                'is_actuals', 'has_data_leakage'  # New columns
            ] + day_cols)

        # Forecast actuals table
        if os.path.exists(self.forecast_actuals_path):
            self.forecast_actuals = pd.read_parquet(self.forecast_actuals_path)
        else:
            self.forecast_actuals = pd.DataFrame(columns=[
                'forecast_date', 'commodity', 'actual_close'
            ])

    def write_point_forecasts(self,
                             forecast_df: pd.DataFrame,
                             model_version: str,
                             commodity: str,
                             data_cutoff_date: datetime,
                             generation_timestamp: Optional[datetime] = None,
                             prediction_intervals: Optional[Dict] = None,
                             model_success: bool = True,
                             actuals_df: Optional[pd.DataFrame] = None) -> int:
        """
        Write point forecasts to production table with actuals and data quality flags.

        Args:
            forecast_df: DataFrame with columns ['date', 'forecast']
            model_version: Model identifier (e.g., 'sarimax_v1')
            commodity: 'Coffee' or 'Sugar'
            data_cutoff_date: Last date in training data
            generation_timestamp: When forecast was generated
            prediction_intervals: Dict with 'lower_95', 'upper_95', 'std'
            model_success: Did model converge successfully?
            actuals_df: Optional DataFrame with columns ['date', 'actual'] for backfill

        Returns:
            Number of rows written

        Schema:
            One row per (forecast_date, model_version, day_ahead)
            New columns:
            - actual_close: Realized price (NULL if future date)
            - has_data_leakage: True if forecast_date <= data_cutoff_date (should be 0!)

        Example:
            forecast_date | day_ahead | forecast_mean | actual_close | has_data_leakage
            2024-01-15    | 1         | 185.5        | 185.2       | False
            2024-01-16    | 2         | 186.2        | NULL        | False
        """
        if generation_timestamp is None:
            generation_timestamp = datetime.now()

        # Create actuals lookup dictionary if provided
        actuals_dict = {}
        if actuals_df is not None:
            actuals_df = actuals_df.copy()
            actuals_df['date'] = pd.to_datetime(actuals_df['date'])

            # Handle both 'actual' and 'close' column names
            actual_col = 'actual' if 'actual' in actuals_df.columns else 'close'
            actuals_dict = dict(zip(actuals_df['date'], actuals_df[actual_col]))

        rows = []

        for idx, row in forecast_df.iterrows():
            forecast_date = pd.to_datetime(row['date'])
            day_ahead = (forecast_date - data_cutoff_date).days

            # Get prediction intervals if available
            if prediction_intervals:
                lower_95 = prediction_intervals.get('lower_95', [None] * len(forecast_df))[idx]
                upper_95 = prediction_intervals.get('upper_95', [None] * len(forecast_df))[idx]
                forecast_std = prediction_intervals.get('std', [None] * len(forecast_df))[idx]
            else:
                lower_95 = None
                upper_95 = None
                forecast_std = None

            # Lookup actual if available
            actual_close = actuals_dict.get(forecast_date, None)

            # Set data leakage flag
            has_data_leakage = forecast_date <= data_cutoff_date

            rows.append({
                'forecast_date': forecast_date,
                'data_cutoff_date': data_cutoff_date,
                'generation_timestamp': generation_timestamp,
                'day_ahead': day_ahead,
                'forecast_mean': round(row['forecast'], 2),
                'forecast_std': round(forecast_std, 2) if forecast_std is not None else None,
                'lower_95': round(lower_95, 2) if lower_95 is not None else None,
                'upper_95': round(upper_95, 2) if upper_95 is not None else None,
                'model_version': model_version,
                'commodity': commodity,
                'model_success': model_success,
                'actual_close': round(actual_close, 2) if actual_close is not None else None,
                'has_data_leakage': has_data_leakage
            })

        # Append to table
        new_rows_df = pd.DataFrame(rows)
        self.point_forecasts = pd.concat([self.point_forecasts, new_rows_df], ignore_index=True)

        # Save
        self.point_forecasts.to_parquet(self.point_forecasts_path, index=False)

        print(f"✓ Wrote {len(rows)} point forecasts for {commodity} ({model_version})")
        print(f"  Forecast dates: {forecast_df['date'].min()} to {forecast_df['date'].max()}")
        print(f"  Day ahead range: 1 to {day_ahead}")

        return len(rows)

    def write_distributions(self,
                           forecast_start_date: datetime,
                           data_cutoff_date: datetime,
                           model_version: str,
                           commodity: str,
                           sample_paths: np.ndarray,
                           generation_timestamp: Optional[datetime] = None,
                           n_paths: int = 2000,
                           actuals_df: Optional[pd.DataFrame] = None) -> int:
        """
        Write Monte Carlo distributions to production table with actuals as path 0.

        Args:
            forecast_start_date: First day of forecast
            data_cutoff_date: Last date in training data
            model_version: Model identifier
            commodity: 'Coffee' or 'Sugar'
            sample_paths: Array of shape (n_paths, 14) with forecasted prices
            generation_timestamp: When generated
            n_paths: Number of sample paths (default 2000)
            actuals_df: Optional DataFrame with columns ['date', 'actual'] for backfill
                        If provided, actuals will be stored as path_id=0

        Returns:
            Number of rows written

        Schema:
            One row per (forecast_start_date, model_version, path_id)
            Columns: path_id, forecast_start_date, ..., day_1, day_2, ..., day_14
            New columns:
            - is_actuals: True for path_id=0 (actuals), False for forecast paths
            - has_data_leakage: True if forecast_start_date <= data_cutoff_date

        Purpose:
            Monte Carlo paths for risk analysis (VaR, CVaR)
            path_id=0 reserved for actuals when available
        """
        if generation_timestamp is None:
            generation_timestamp = datetime.now()

        if sample_paths.shape[1] != 14:
            raise ValueError(f"Expected 14-day forecasts, got {sample_paths.shape[1]}")

        rows = []

        # Set data leakage flag
        has_data_leakage = forecast_start_date <= data_cutoff_date

        # Create actuals lookup if provided
        actuals_dict = {}
        if actuals_df is not None:
            actuals_df = actuals_df.copy()
            actuals_df['date'] = pd.to_datetime(actuals_df['date'])

            # Handle both 'actual' and 'close' column names
            actual_col = 'actual' if 'actual' in actuals_df.columns else 'close'
            actuals_dict = dict(zip(actuals_df['date'], actuals_df[actual_col]))

        # If actuals provided, create path 0 with actuals
        if actuals_dict:
            actuals_row = {
                'path_id': 0,
                'forecast_start_date': forecast_start_date,
                'data_cutoff_date': data_cutoff_date,
                'generation_timestamp': generation_timestamp,
                'model_version': model_version,
                'commodity': commodity,
                'is_actuals': True,
                'has_data_leakage': has_data_leakage
            }

            # Add actuals for each day (or NULL if not available)
            for day in range(1, 15):
                day_date = forecast_start_date + timedelta(days=day - 1)
                actual_value = actuals_dict.get(day_date, None)
                actuals_row[f'day_{day}'] = round(actual_value, 2) if actual_value is not None else None

            rows.append(actuals_row)

        # Add forecast paths starting from path_id=1 (or 0 if no actuals)
        start_path_id = 1 if actuals_dict else 0
        for path_id in range(min(n_paths, sample_paths.shape[0])):
            row = {
                'path_id': start_path_id + path_id,
                'forecast_start_date': forecast_start_date,
                'data_cutoff_date': data_cutoff_date,
                'generation_timestamp': generation_timestamp,
                'model_version': model_version,
                'commodity': commodity,
                'is_actuals': False,
                'has_data_leakage': has_data_leakage
            }

            # Add day_1 to day_14 columns (rounded to 2 decimals)
            for day in range(1, 15):
                row[f'day_{day}'] = round(sample_paths[path_id, day - 1], 2)

            rows.append(row)

        # Append to table
        new_rows_df = pd.DataFrame(rows)
        self.distributions = pd.concat([self.distributions, new_rows_df], ignore_index=True)

        # Save
        self.distributions.to_parquet(self.distributions_path, index=False)

        print(f"✓ Wrote {len(rows)} distribution paths for {commodity} ({model_version})")
        print(f"  Forecast start: {forecast_start_date}")
        print(f"  Paths: {len(rows)}")

        return len(rows)

    def write_forecast_actuals(self,
                              actuals_df: pd.DataFrame,
                              commodity: str) -> int:
        """
        Write realized prices for backtesting.

        Args:
            actuals_df: DataFrame with columns ['date', 'actual'] or ['date', 'close']
            commodity: 'Coffee' or 'Sugar'

        Returns:
            Number of rows written

        Schema:
            One row per (forecast_date, commodity)
            Columns: forecast_date, commodity, actual_close

        Purpose:
            Store realized prices for backtesting and evaluation
        """
        # Rename columns
        if 'actual' in actuals_df.columns:
            actuals_df = actuals_df.rename(columns={'actual': 'actual_close'})
        elif 'close' in actuals_df.columns:
            actuals_df = actuals_df.rename(columns={'close': 'actual_close'})

        rows = []

        for idx, row in actuals_df.iterrows():
            rows.append({
                'forecast_date': pd.to_datetime(row['date']),
                'commodity': commodity,
                'actual_close': round(row['actual_close'], 2)
            })

        # Append to table
        new_rows_df = pd.DataFrame(rows)

        # Remove duplicates (keep latest)
        combined = pd.concat([self.forecast_actuals, new_rows_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=['forecast_date', 'commodity'], keep='last')
        self.forecast_actuals = combined

        # Save
        self.forecast_actuals.to_parquet(self.forecast_actuals_path, index=False)

        print(f"✓ Wrote {len(rows)} forecast actuals for {commodity}")
        print(f"  Date range: {actuals_df['date'].min()} to {actuals_df['date'].max()}")

        return len(rows)

    def get_latest_forecast(self, commodity: str, model_version: str) -> Optional[pd.DataFrame]:
        """
        Get latest forecast for a commodity and model.

        Returns:
            DataFrame with point forecasts or None
        """
        df = self.point_forecasts[
            (self.point_forecasts['commodity'] == commodity) &
            (self.point_forecasts['model_version'] == model_version)
        ].copy()

        if len(df) == 0:
            return None

        # Get most recent generation
        latest_gen = df['generation_timestamp'].max()
        df = df[df['generation_timestamp'] == latest_gen]

        return df.sort_values('forecast_date')

    def export_for_trading_agent(self,
                                commodity: str,
                                model_version: str,
                                output_path: str = "trading_agent_forecast.json"):
        """
        Export latest forecast in format for trading agent.

        Args:
            commodity: 'Coffee' or 'Sugar'
            model_version: Model identifier
            output_path: Path to JSON file
        """
        import json

        forecast_df = self.get_latest_forecast(commodity, model_version)

        if forecast_df is None:
            print(f"✗ No forecast found for {commodity} ({model_version})")
            return None

        # Format for trading agent
        trading_forecast = {
            'commodity': commodity,
            'model': model_version,
            'generation_timestamp': str(forecast_df['generation_timestamp'].iloc[0]),
            'data_cutoff_date': str(forecast_df['data_cutoff_date'].iloc[0]),
            'horizon_days': len(forecast_df),
            'forecasts': []
        }

        # Build forecast entries
        for _, row in forecast_df.iterrows():
            entry = {
                'date': str(row['forecast_date'].date()),
                'day_ahead': int(row['day_ahead']),
                'forecast': float(row['forecast_mean']),
                'lower_95': float(row['lower_95']) if pd.notna(row['lower_95']) else None,
                'upper_95': float(row['upper_95']) if pd.notna(row['upper_95']) else None,
                'forecast_std': float(row['forecast_std']) if pd.notna(row['forecast_std']) else None
            }
            trading_forecast['forecasts'].append(entry)

        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(trading_forecast, f, indent=2)

        print(f"✓ Exported forecast to {output_path}")
        print(f"  Model: {model_version}")
        print(f"  Generation: {forecast_df['generation_timestamp'].iloc[0]}")
        print(f"  Horizon: {len(forecast_df)} days")

        return trading_forecast


# Example production usage
EXAMPLE_PRODUCTION_USAGE = """
# PRODUCTION DEPLOYMENT WORKFLOW
# ================================

from ground_truth.storage.production_writer import ProductionForecastWriter
from ground_truth.storage.model_metadata_schema import ModelMetadataStore
from ground_truth.models import sarimax
import pandas as pd
from datetime import datetime

# 1. Load full historical data
df = pd.read_parquet("unified_data_with_gdelt.parquet")
df_coffee = df[df['commodity'] == 'Coffee'].set_index('date').sort_index()

# 2. Get best model from metadata store (based on walk-forward evaluation)
model_store = ModelMetadataStore("model_registry.parquet")
best_model = model_store.get_best_model(
    commodity='Coffee',
    metric='directional_accuracy_from_day0'  # Use Dir Day0 for trading!
)

print(f"Best model: {best_model['model_name']}")
print(f"  MAE: ${best_model['mae']:.2f}")
print(f"  Dir Day0: {best_model.get('directional_accuracy_from_day0', 'N/A')}%")

# 3. Train on FULL history (no train/test split in production)
print(f"Training on full history ({len(df_coffee)} days)...")

data_cutoff_date = df_coffee.index[-1]
generation_timestamp = datetime.now()

result = sarimax.sarimax_forecast_with_metadata(
    df_pandas=df_coffee,
    commodity='Coffee',
    target='close',
    horizon=14,
    exog_features=['temp_c', 'humidity_pct', 'precipitation_mm']
)

# 4. Write to production tables
writer = ProductionForecastWriter("production_forecasts")

# Write point forecasts (one row per day_ahead)
writer.write_point_forecasts(
    forecast_df=result['forecast_df'],
    model_version='sarimax_weather_v1',
    commodity='Coffee',
    data_cutoff_date=data_cutoff_date,
    generation_timestamp=generation_timestamp,
    prediction_intervals=None,  # TODO: Add prediction intervals
    model_success=True
)

# Generate Monte Carlo paths for distributions table
# TODO: Implement bootstrap or simulation for sample paths
# For now, we can generate simple paths from residuals
n_paths = 2000
forecast_values = result['forecast_df']['forecast'].values
sample_paths = np.random.normal(
    loc=forecast_values,
    scale=2.5,  # Residual std from walk-forward evaluation
    size=(n_paths, 14)
)

writer.write_distributions(
    forecast_start_date=result['forecast_df']['date'].iloc[0],
    data_cutoff_date=data_cutoff_date,
    model_version='sarimax_weather_v1',
    commodity='Coffee',
    sample_paths=sample_paths,
    generation_timestamp=generation_timestamp
)

# 5. Export for trading agent
writer.export_for_trading_agent(
    commodity='Coffee',
    model_version='sarimax_weather_v1',
    output_path='trading_agent_forecast.json'
)

print("✅ Production forecasts stored successfully!")
print(f"   Point forecasts: {writer.point_forecasts_path}")
print(f"   Distributions: {writer.distributions_path}")

# WEEKLY RETRAINING SCRIPT
# =========================
# Run this script every Monday morning:
# 1. Load latest data (including last week's actuals)
# 2. Write last week's actuals to forecast_actuals table
# 3. Retrain model on full history
# 4. Generate new 14-day forecast
# 5. Store in production tables
# 6. Export for trading agent
# 7. Monitor: Compare last week's forecast vs actuals
"""
