"""
Temporal Fusion Transformer (TFT) model for commodity price forecasting.

TFT is a state-of-the-art deep learning model designed specifically for
multi-horizon time series forecasting with interpretable attention mechanisms.

Key Features:
- Multi-horizon forecasting (1-14 days ahead)
- Automatic feature importance via attention
- Probabilistic forecasts (quantiles for uncertainty)
- Handles multiple covariates (weather, GDELT, VIX, etc.)
- Variable selection network learns which features matter

Reference:
"Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
Lim et al., 2021, https://arxiv.org/abs/1912.09363
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import timedelta
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')


def prepare_tft_data(
    df_pandas: pd.DataFrame,
    horizon: int = 14,
    max_encoder_length: int = 60,
    target: str = 'close',
    time_varying_known_reals: Optional[List[str]] = None,
    time_varying_unknown_reals: Optional[List[str]] = None
) -> TimeSeriesDataSet:
    """
    Prepare data for Temporal Fusion Transformer.

    Args:
        df_pandas: DataFrame with time series data (must have 'date' index)
        horizon: Forecast horizon (days ahead)
        max_encoder_length: Max lookback window
        target: Target variable name
        time_varying_known_reals: Known future covariates (e.g., weather forecasts)
        time_varying_unknown_reals: Unknown future covariates (e.g., lagged prices)

    Returns:
        TimeSeriesDataSet ready for TFT training
    """
    # Reset index to make date a column
    df = df_pandas.reset_index()

    # Create time index (days since start)
    df['time_idx'] = (df['date'] - df['date'].min()).dt.days

    # Add group identifier (single time series)
    df['series_id'] = 'coffee'  # Could extend to multi-commodity

    # Default covariates if not specified
    if time_varying_known_reals is None:
        time_varying_known_reals = []

    if time_varying_unknown_reals is None:
        time_varying_unknown_reals = []

    # Add target to unknown reals (we don't know future prices)
    if target not in time_varying_unknown_reals:
        time_varying_unknown_reals.append(target)

    # Create TimeSeriesDataSet
    training = TimeSeriesDataSet(
        df,
        time_idx='time_idx',
        target=target,
        group_ids=['series_id'],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=horizon,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        target_normalizer=GroupNormalizer(
            groups=['series_id'], transformation='softplus'
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,  # Handle weekends/holidays in financial data
    )

    return training


def tft_forecast_with_metadata(
    df_pandas: pd.DataFrame,
    commodity: str,
    target: str = 'close',
    horizon: int = 14,
    max_encoder_length: int = 60,
    hidden_size: int = 32,
    attention_head_size: int = 4,
    dropout: float = 0.1,
    learning_rate: float = 0.001,
    max_epochs: int = 50,
    batch_size: int = 32,
    gradient_clip_val: float = 0.1,
    exog_features: Optional[List[str]] = None,
    **kwargs
) -> Dict:
    """
    Generate forecast using Temporal Fusion Transformer.

    Args:
        df_pandas: Training data with datetime index
        commodity: Commodity name (e.g., 'Coffee')
        target: Target variable to forecast
        horizon: Forecast horizon (days)
        max_encoder_length: Lookback window (days)
        hidden_size: LSTM hidden size (smaller = faster)
        attention_head_size: Number of attention heads
        dropout: Dropout rate
        learning_rate: Adam learning rate
        max_epochs: Max training epochs
        batch_size: Training batch size
        gradient_clip_val: Gradient clipping threshold
        exog_features: List of exogenous features (weather, GDELT, etc.)

    Returns:
        Dictionary with:
            - forecast_df: Point forecasts
            - quantiles: Probabilistic forecasts (10th, 50th, 90th percentiles)
            - std: Forecast uncertainty
            - attention_weights: Feature importance (interpretability)
    """
    try:
        # Separate known (weather) vs unknown (price lags) features
        time_varying_known_reals = []
        time_varying_unknown_reals = []

        if exog_features:
            # Weather features are "known" (we can forecast them)
            weather_features = ['temp_max_c', 'temp_min_c', 'temp_mean_c',
                              'precipitation_mm', 'humidity_mean_pct']

            for feat in exog_features:
                if any(w in feat for w in weather_features):
                    time_varying_known_reals.append(feat)
                else:
                    # GDELT, VIX, etc. are "unknown" (we don't know future values)
                    time_varying_unknown_reals.append(feat)

        # Prepare data
        training_data = prepare_tft_data(
            df_pandas=df_pandas,
            horizon=horizon,
            max_encoder_length=max_encoder_length,
            target=target,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_reals=time_varying_unknown_reals
        )

        # Create validation set (last 10% of data)
        validation = TimeSeriesDataSet.from_dataset(
            training_data,
            df_pandas.reset_index().assign(
                time_idx=lambda x: (x['date'] - x['date'].min()).dt.days,
                series_id='coffee'
            ),
            min_prediction_idx=training_data.index.time.max() - horizon
        )

        # Create dataloaders
        train_dataloader = training_data.to_dataloader(
            train=True, batch_size=batch_size, num_workers=0
        )
        val_dataloader = validation.to_dataloader(
            train=False, batch_size=batch_size, num_workers=0
        )

        # Initialize model
        tft = TemporalFusionTransformer.from_dataset(
            training_data,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            attention_head_size=attention_head_size,
            dropout=dropout,
            hidden_continuous_size=8,
            loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),  # Probabilistic forecasts
            reduce_on_plateau_patience=3,
            optimizer='adam'
        )

        # Train model
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=5,
            mode='min'
        )

        trainer = Trainer(
            max_epochs=max_epochs,
            accelerator='cpu',  # Use 'gpu' if available
            gradient_clip_val=gradient_clip_val,
            callbacks=[early_stop_callback],
            enable_progress_bar=False,
            logger=False
        )

        trainer.fit(
            tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )

        # Generate predictions on last window
        raw_predictions = tft.predict(
            val_dataloader,
            mode='quantiles',
            return_x=True
        )

        # Extract quantiles (10th, 50th, 90th percentiles)
        predictions = raw_predictions[0].cpu().numpy()  # Shape: (batch, horizon, 3)

        # Get last prediction (most recent forecast)
        last_pred = predictions[-1]  # Shape: (horizon, 3)

        # Create forecast DataFrame
        last_date = df_pandas.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=horizon,
            freq='D'
        )

        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'forecast': last_pred[:, 1],  # Median (50th percentile)
            'lower_10': last_pred[:, 0],  # 10th percentile
            'upper_90': last_pred[:, 2],  # 90th percentile
        })
        forecast_df = forecast_df.set_index('date')

        # Calculate std from quantiles (approximate)
        # 80% confidence interval ≈ 1.28 std on each side
        forecast_std = (forecast_df['upper_90'] - forecast_df['lower_10']) / (2 * 1.28)

        # Get feature importance from attention weights
        interpretation = tft.interpret_output(
            raw_predictions[0],
            reduction='sum'
        )

        attention_weights = {
            'encoder_attention': interpretation.get('attention', None),
            'variable_selection': interpretation.get('variable_selection', None)
        }

        return {
            'forecast_df': forecast_df,
            'quantiles': {
                '10': forecast_df['lower_10'].values,
                '50': forecast_df['forecast'].values,
                '90': forecast_df['upper_90'].values
            },
            'std': forecast_std.mean(),  # Average uncertainty
            'attention_weights': attention_weights,
            'model': tft,  # Return trained model for inspection
            'error': None
        }

    except Exception as e:
        # Return error with fallback forecast
        print(f"TFT Error: {e}")

        # Fallback to naive forecast
        last_value = df_pandas[target].iloc[-1]
        forecast_dates = pd.date_range(
            start=df_pandas.index[-1] + timedelta(days=1),
            periods=horizon,
            freq='D'
        )

        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'forecast': last_value
        }).set_index('date')

        return {
            'forecast_df': forecast_df,
            'quantiles': None,
            'std': 0.0,
            'attention_weights': None,
            'model': None,
            'error': str(e)
        }


def tft_ensemble_forecast(
    df_pandas: pd.DataFrame,
    commodity: str,
    n_models: int = 5,
    **kwargs
) -> Dict:
    """
    Ensemble of TFT models with different initializations.

    Trains multiple TFT models and averages their predictions
    for improved robustness.

    Args:
        df_pandas: Training data
        commodity: Commodity name
        n_models: Number of models in ensemble
        **kwargs: Arguments passed to tft_forecast_with_metadata

    Returns:
        Dictionary with ensemble forecasts and uncertainty
    """
    forecasts = []
    stds = []

    for i in range(n_models):
        # Set random seed for reproducibility
        torch.manual_seed(42 + i)
        np.random.seed(42 + i)

        result = tft_forecast_with_metadata(df_pandas, commodity, **kwargs)

        if result['error'] is None:
            forecasts.append(result['forecast_df']['forecast'].values)
            stds.append(result['std'])

    if len(forecasts) == 0:
        # All models failed, return error
        return tft_forecast_with_metadata(df_pandas, commodity, **kwargs)

    # Average predictions
    ensemble_forecast = np.mean(forecasts, axis=0)
    ensemble_std = np.mean(stds)

    # Create forecast DataFrame
    last_date = df_pandas.index[-1]
    horizon = len(ensemble_forecast)
    forecast_dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=horizon,
        freq='D'
    )

    forecast_df = pd.DataFrame({
        'date': forecast_dates,
        'forecast': ensemble_forecast
    }).set_index('date')

    return {
        'forecast_df': forecast_df,
        'quantiles': None,
        'std': ensemble_std,
        'attention_weights': None,
        'model': None,
        'error': None,
        'n_models': len(forecasts)
    }
