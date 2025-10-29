"""Model metadata storage schema for trading agent.

Stores model versions, performance metrics, and configurations.
"""

import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Optional
import os


class ModelMetadataStore:
    """
    Stores and retrieves model metadata for trading agent selection.

    Schema:
        - model_id: Unique identifier (model_name + timestamp)
        - model_name: Human-readable name
        - version: Version string (e.g., "v1.0")
        - commodity: Coffee, Sugar, etc.
        - trained_date: When model was trained
        - train_start: Training period start
        - train_end: Training period end
        - forecast_horizon: Days ahead
        - parameters: JSON of model parameters
        - metrics: Performance metrics (MAE, RMSE, etc.)
        - status: active, deprecated, testing
        - model_path: Path to saved model artifact
        - feature_list: List of required features
        - dependencies: Required packages/versions
    """

    def __init__(self, storage_path: str = "model_registry.parquet"):
        """
        Initialize metadata store.

        Args:
            storage_path: Path to parquet file for storage
        """
        self.storage_path = storage_path

        # Initialize empty registry if doesn't exist
        if not os.path.exists(storage_path):
            self._create_empty_registry()
        else:
            self.registry = pd.read_parquet(storage_path)

    def _create_empty_registry(self):
        """Create empty registry with schema."""
        self.registry = pd.DataFrame(columns=[
            'model_id',
            'model_name',
            'version',
            'commodity',
            'trained_date',
            'train_start',
            'train_end',
            'forecast_horizon',
            'parameters',
            'mae',
            'rmse',
            'mape',
            'directional_accuracy',
            'directional_accuracy_from_day0',
            'status',
            'model_path',
            'feature_list',
            'dependencies',
            'notes'
        ])

    def register_model(self, model_result: Dict, status: str = 'active',
                      model_path: Optional[str] = None) -> str:
        """
        Register a new model in the metadata store.

        Args:
            model_result: Output from model training (with forecast_df, metrics, etc.)
            status: active, testing, deprecated
            model_path: Path to saved model artifact

        Returns:
            model_id: Unique identifier for this model
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"{model_result['model_name'].replace(' ', '_')}_{timestamp}"

        # Extract metrics
        metrics = model_result.get('metrics', {})

        # Build row
        row = {
            'model_id': model_id,
            'model_name': model_result['model_name'],
            'version': timestamp,
            'commodity': model_result['commodity'],
            'trained_date': datetime.now(),
            'train_start': model_result.get('training_start', None),
            'train_end': model_result.get('training_end', None),
            'forecast_horizon': len(model_result['forecast_df']),
            'parameters': json.dumps(model_result.get('parameters', {})),
            'mae': metrics.get('mae', None),
            'rmse': metrics.get('rmse', None),
            'mape': metrics.get('mape', None),
            'directional_accuracy': metrics.get('directional_accuracy', None),
            'directional_accuracy_from_day0': metrics.get('directional_accuracy_from_day0', None),
            'status': status,
            'model_path': model_path,
            'feature_list': json.dumps(model_result.get('parameters', {}).get('exog_features', [])),
            'dependencies': json.dumps(self._get_dependencies()),
            'notes': ''
        }

        # Append to registry
        self.registry = pd.concat([self.registry, pd.DataFrame([row])], ignore_index=True)

        # Save
        self.save()

        return model_id

    def _get_dependencies(self) -> Dict:
        """Get current package versions."""
        import xgboost
        import pandas
        import numpy

        return {
            'xgboost': xgboost.__version__,
            'pandas': pandas.__version__,
            'numpy': numpy.__version__
        }

    def get_active_models(self, commodity: str = None) -> pd.DataFrame:
        """
        Get all active models, optionally filtered by commodity.

        Args:
            commodity: Filter by commodity (None = all)

        Returns:
            DataFrame of active models sorted by MAE
        """
        df = self.registry[self.registry['status'] == 'active'].copy()

        if commodity:
            df = df[df['commodity'] == commodity]

        # Sort by MAE (best first)
        df = df.sort_values('mae')

        return df

    def get_best_model(self, commodity: str, metric: str = 'mae') -> Dict:
        """
        Get best model for a commodity based on metric.

        Args:
            commodity: Commodity to forecast
            metric: Metric to optimize (mae, rmse, directional_accuracy)

        Returns:
            Dict with model metadata
        """
        df = self.get_active_models(commodity)

        if len(df) == 0:
            return None

        # Sort by metric (ascending for errors, descending for accuracy)
        if metric in ['mae', 'rmse', 'mape']:
            best = df.sort_values(metric).iloc[0]
        else:
            best = df.sort_values(metric, ascending=False).iloc[0]

        return best.to_dict()

    def update_model_status(self, model_id: str, status: str):
        """
        Update model status (active, testing, deprecated).

        Args:
            model_id: Model identifier
            status: New status
        """
        self.registry.loc[self.registry['model_id'] == model_id, 'status'] = status
        self.save()

    def get_model_by_id(self, model_id: str) -> Optional[Dict]:
        """
        Get model metadata by ID.

        Args:
            model_id: Model identifier

        Returns:
            Dict with model metadata or None
        """
        matches = self.registry[self.registry['model_id'] == model_id]

        if len(matches) == 0:
            return None

        return matches.iloc[0].to_dict()

    def save(self):
        """Save registry to disk."""
        self.registry.to_parquet(self.storage_path, index=False)

    def export_for_trading_agent(self, output_path: str = "trading_agent_models.json"):
        """
        Export active models in format for trading agent.

        Args:
            output_path: Path to JSON file
        """
        active = self.get_active_models()

        # Convert to trading agent format
        models_for_agent = []

        for _, row in active.iterrows():
            models_for_agent.append({
                'model_id': row['model_id'],
                'model_name': row['model_name'],
                'commodity': row['commodity'],
                'forecast_horizon': int(row['forecast_horizon']),
                'mae': float(row['mae']),
                'rmse': float(row['rmse']),
                'mape': float(row['mape']),
                'directional_accuracy': float(row['directional_accuracy']) if pd.notna(row['directional_accuracy']) else None,
                'directional_accuracy_from_day0': float(row['directional_accuracy_from_day0']) if pd.notna(row['directional_accuracy_from_day0']) else None,
                'model_path': row['model_path'],
                'features_required': json.loads(row['feature_list']),
                'parameters': json.loads(row['parameters'])
            })

        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(models_for_agent, f, indent=2)

        print(f"✓ Exported {len(models_for_agent)} models to {output_path}")

        return models_for_agent


# Example usage
EXAMPLE_USAGE = """
from ground_truth.storage.model_metadata_schema import ModelMetadataStore

# Initialize store
store = ModelMetadataStore("model_registry.parquet")

# Register a model after training
model_result = {
    'model_name': 'XGBoost+Weather',
    'commodity': 'Coffee',
    'forecast_df': forecast_df,
    'metrics': {
        'mae': 1.99,
        'rmse': 2.37,
        'mape': 1.08,
        'directional_accuracy': 38.5,
        'directional_accuracy_from_day0': 45.2
    },
    'parameters': {
        'lags': [1, 7, 14],
        'exog_features': ['temp_c', 'humidity_pct']
    },
    'training_end': pd.Timestamp('2023-12-31')
}

model_id = store.register_model(model_result, status='active')

# Get best model for trading agent
best = store.get_best_model('Coffee', metric='mae')
print(f"Best model: {best['model_name']} (MAE: ${best['mae']:.2f})")

# Export for trading agent
store.export_for_trading_agent("trading_agent_models.json")

# Trading agent can then load models:
import json
with open("trading_agent_models.json") as f:
    available_models = json.load(f)

# Select model based on criteria
best_directional = max(available_models,
                      key=lambda x: x['directional_accuracy_from_day0'])
"""
