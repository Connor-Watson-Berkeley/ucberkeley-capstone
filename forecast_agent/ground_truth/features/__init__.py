"""Feature engineering modules for commodity forecasting.

This package provides function-based feature engineering that is:
- Composable: Functions can be chained together
- Reusable: Same function used across multiple models
- Testable: Each function can be unit tested independently
- PySpark-first: Designed for distributed processing
"""

from ground_truth.features import aggregators, covariate_projection, transformers

__all__ = ['aggregators', 'covariate_projection', 'transformers']
