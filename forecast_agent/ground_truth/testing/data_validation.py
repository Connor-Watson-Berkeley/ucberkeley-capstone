"""Data validation tests for unified_data and forecast output tables.

Based on validation queries from forecast v1 notebook.
Tests for duplicates, nulls, data leakage, and schema compliance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime


class UnifiedDataValidator:
    """Validates commodity.silver.unified_data schema and data quality."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize validator with unified data.

        Args:
            df: DataFrame with unified_data schema
        """
        self.df = df
        self.results = {}

    def validate_all(self) -> Dict:
        """Run all validation checks and return results."""
        self.check_schema()
        self.check_duplicates()
        self.check_nulls()
        self.check_data_quality()
        return self.results

    def check_schema(self) -> bool:
        """Verify required columns exist."""
        required_columns = [
            'date', 'is_trading_day', 'commodity', 'close', 'high', 'low', 'open',
            'volume', 'vix', 'region', 'temp_c', 'humidity_pct', 'precipitation_mm'
        ]

        missing = [col for col in required_columns if col not in self.df.columns]

        self.results['schema'] = {
            'passed': len(missing) == 0,
            'missing_columns': missing,
            'total_columns': len(self.df.columns),
            'expected_columns': len(required_columns)
        }

        return len(missing) == 0

    def check_duplicates(self) -> bool:
        """Check for duplicate (date, commodity, region) combinations."""
        total_rows = len(self.df)
        unique_rows = self.df[['date', 'commodity', 'region']].drop_duplicates().shape[0]
        duplicates = total_rows - unique_rows

        self.results['duplicates'] = {
            'passed': duplicates == 0,
            'total_rows': total_rows,
            'unique_rows': unique_rows,
            'duplicate_count': duplicates,
            'duplicate_pct': round(duplicates / total_rows * 100, 2) if total_rows > 0 else 0
        }

        return duplicates == 0

    def check_nulls(self) -> Dict:
        """Check for NULL values in critical columns."""
        critical_columns = ['date', 'commodity', 'close', 'region']
        null_counts = {}

        for col in critical_columns:
            if col in self.df.columns:
                null_count = self.df[col].isna().sum()
                null_counts[col] = {
                    'null_count': int(null_count),
                    'null_pct': round(null_count / len(self.df) * 100, 2)
                }

        # Check all columns for info
        all_null_counts = {}
        for col in self.df.columns:
            null_count = self.df[col].isna().sum()
            if null_count > 0:
                all_null_counts[col] = {
                    'null_count': int(null_count),
                    'null_pct': round(null_count / len(self.df) * 100, 2)
                }

        has_critical_nulls = any(v['null_count'] > 0 for v in null_counts.values())

        self.results['nulls'] = {
            'passed': not has_critical_nulls,
            'critical_columns': null_counts,
            'all_columns_with_nulls': all_null_counts
        }

        return not has_critical_nulls

    def check_data_quality(self) -> Dict:
        """Check for nonsensical values."""
        quality_issues = []

        # Check for negative prices
        if 'close' in self.df.columns:
            negative_prices = (self.df['close'] < 0).sum()
            if negative_prices > 0:
                quality_issues.append(f'Negative close prices: {negative_prices}')

        # Check for unrealistic prices (> $1000/lb for coffee)
        if 'close' in self.df.columns and 'commodity' in self.df.columns:
            coffee_df = self.df[self.df['commodity'] == 'Coffee']
            unrealistic = (coffee_df['close'] > 1000).sum()
            if unrealistic > 0:
                quality_issues.append(f'Unrealistic coffee prices (>$1000): {unrealistic}')

        # Check date range
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'])
            date_range = {
                'min': str(self.df['date'].min().date()),
                'max': str(self.df['date'].max().date()),
                'span_days': (self.df['date'].max() - self.df['date'].min()).days
            }
        else:
            date_range = {}

        # Check for future dates
        if 'date' in self.df.columns:
            future_dates = (self.df['date'] > pd.Timestamp.now()).sum()
            if future_dates > 0:
                quality_issues.append(f'Future dates: {future_dates}')

        self.results['data_quality'] = {
            'passed': len(quality_issues) == 0,
            'issues': quality_issues,
            'date_range': date_range
        }

        return len(quality_issues) == 0

    def print_report(self):
        """Print validation report."""
        print("="*70)
        print("UNIFIED DATA VALIDATION REPORT")
        print("="*70)
        print()

        for check_name, result in self.results.items():
            status = "✓ PASS" if result['passed'] else "✗ FAIL"
            print(f"{status} - {check_name.upper()}")

            # Print details
            for key, value in result.items():
                if key != 'passed':
                    print(f"   {key}: {value}")
            print()


class ForecastOutputValidator:
    """Validates forecast output tables (point_forecasts, distributions)."""

    def __init__(self, point_forecasts: pd.DataFrame = None,
                 distributions: pd.DataFrame = None,
                 actuals: pd.DataFrame = None):
        """
        Initialize validator with forecast tables.

        Args:
            point_forecasts: point_forecasts table
            distributions: distributions table
            actuals: actuals data for comparison
        """
        self.point_forecasts = point_forecasts
        self.distributions = distributions
        self.actuals = actuals
        self.results = {}

    def validate_all(self) -> Dict:
        """Run all validation checks."""
        if self.point_forecasts is not None:
            self.check_point_forecasts_schema()
            self.check_data_leakage()
            self.check_point_forecasts_quality()

        if self.distributions is not None:
            self.check_distributions_schema()
            self.check_distribution_paths()

        return self.results

    def check_point_forecasts_schema(self) -> bool:
        """Verify point_forecasts schema."""
        required_columns = [
            'forecast_date', 'data_cutoff_date', 'generation_timestamp',
            'day_ahead', 'forecast_mean', 'forecast_std', 'lower_95', 'upper_95',
            'model_version', 'commodity', 'model_success',
            'actual_close', 'has_data_leakage'  # New columns
        ]

        missing = [col for col in required_columns if col not in self.point_forecasts.columns]

        self.results['point_forecasts_schema'] = {
            'passed': len(missing) == 0,
            'missing_columns': missing
        }

        return len(missing) == 0

    def check_data_leakage(self) -> bool:
        """Check for data leakage (forecast_date <= data_cutoff_date)."""
        if self.point_forecasts is None:
            return True

        df = self.point_forecasts.copy()
        df['forecast_date'] = pd.to_datetime(df['forecast_date'])
        df['data_cutoff_date'] = pd.to_datetime(df['data_cutoff_date'])

        # Data leakage = forecast_date <= data_cutoff_date
        leakage_count = (df['forecast_date'] <= df['data_cutoff_date']).sum()
        total = len(df)

        # Check if has_data_leakage flag is correct
        if 'has_data_leakage' in df.columns:
            flag_correct = ((df['forecast_date'] <= df['data_cutoff_date']) == df['has_data_leakage']).all()
        else:
            flag_correct = False

        self.results['data_leakage'] = {
            'passed': leakage_count == 0,
            'leakage_count': int(leakage_count),
            'total_forecasts': int(total),
            'leakage_pct': round(leakage_count / total * 100, 2) if total > 0 else 0,
            'flag_correct': flag_correct
        }

        return leakage_count == 0

    def check_point_forecasts_quality(self) -> Dict:
        """Check forecast quality and flags."""
        if self.point_forecasts is None:
            return {}

        df = self.point_forecasts
        issues = []

        # Check for negative forecasts
        negative = (df['forecast_mean'] < 0).sum()
        if negative > 0:
            issues.append(f'Negative forecasts: {negative}')

        # Check for NULL forecasts where model_success=True
        if 'model_success' in df.columns:
            null_forecasts_success = df[df['model_success'] == True]['forecast_mean'].isna().sum()
            if null_forecasts_success > 0:
                issues.append(f'NULL forecasts despite success: {null_forecasts_success}')

        # Check day_ahead calculation
        if 'day_ahead' in df.columns and 'forecast_date' in df.columns and 'data_cutoff_date' in df.columns:
            df['forecast_date'] = pd.to_datetime(df['forecast_date'])
            df['data_cutoff_date'] = pd.to_datetime(df['data_cutoff_date'])
            df['calculated_day_ahead'] = (df['forecast_date'] - df['data_cutoff_date']).dt.days
            incorrect_day_ahead = (df['day_ahead'] != df['calculated_day_ahead']).sum()
            if incorrect_day_ahead > 0:
                issues.append(f'Incorrect day_ahead calculation: {incorrect_day_ahead} rows')

        self.results['point_forecasts_quality'] = {
            'passed': len(issues) == 0,
            'issues': issues
        }

        return len(issues) == 0

    def check_distributions_schema(self) -> bool:
        """Verify distributions schema."""
        required_columns = [
            'path_id', 'forecast_start_date', 'data_cutoff_date',
            'generation_timestamp', 'model_version', 'commodity',
            'is_actuals', 'has_data_leakage'  # New columns
        ]

        # Plus day_1 to day_14
        day_columns = [f'day_{i}' for i in range(1, 15)]
        required_columns.extend(day_columns)

        missing = [col for col in required_columns if col not in self.distributions.columns]

        self.results['distributions_schema'] = {
            'passed': len(missing) == 0,
            'missing_columns': missing
        }

        return len(missing) == 0

    def check_distribution_paths(self) -> Dict:
        """Check distribution path consistency."""
        if self.distributions is None:
            return {}

        df = self.distributions
        issues = []

        # Check is_actuals flag consistency (should be True only for path_id=0)
        if 'is_actuals' in df.columns and 'path_id' in df.columns:
            # is_actuals=True should only be for path_id=0
            incorrect_flags = df[(df['is_actuals'] == True) & (df['path_id'] != 0)].shape[0]
            if incorrect_flags > 0:
                issues.append(f'is_actuals=True for non-zero path_id: {incorrect_flags} rows')

            # path_id=0 should have is_actuals=True
            path_0_rows = df[df['path_id'] == 0]
            if len(path_0_rows) > 0:
                incorrect_path_0 = (path_0_rows['is_actuals'] == False).sum()
                if incorrect_path_0 > 0:
                    issues.append(f'path_id=0 with is_actuals=False: {incorrect_path_0} rows')

        # Check for expected number of paths per forecast_start_date
        # (should be 2000 forecast paths + 1 actuals path = 2001, or 2000 if no actuals)
        path_counts = df.groupby('forecast_start_date')['path_id'].nunique()
        incorrect_counts = ((path_counts != 2000) & (path_counts != 2001)).sum()
        if incorrect_counts > 0:
            issues.append(f'Forecast dates with unexpected path counts: {incorrect_counts}')

        # Check for negative prices in any path
        day_columns = [f'day_{i}' for i in range(1, 15)]
        for col in day_columns:
            if col in df.columns:
                negative = (df[col] < 0).sum()
                if negative > 0:
                    issues.append(f'Negative prices in {col}: {negative}')

        # Check path_id range (should start at 0 or 1 depending on actuals)
        if 'path_id' in df.columns:
            min_path = df['path_id'].min()
            max_path = df['path_id'].max()
            # Valid ranges: 0-2000 (with actuals) or 1-2000 (without actuals) or 0-N for partial data
            if min_path < 0:
                issues.append(f'Negative path_id found: min={min_path}')

        self.results['distribution_paths'] = {
            'passed': len(issues) == 0,
            'issues': issues,
            'unique_forecast_dates': int(df['forecast_start_date'].nunique()) if 'forecast_start_date' in df.columns else 0,
            'total_paths': int(len(df))
        }

        return len(issues) == 0

    def print_report(self):
        """Print validation report."""
        print("="*70)
        print("FORECAST OUTPUT VALIDATION REPORT")
        print("="*70)
        print()

        for check_name, result in self.results.items():
            status = "✓ PASS" if result['passed'] else "✗ FAIL"
            print(f"{status} - {check_name.upper()}")

            for key, value in result.items():
                if key != 'passed':
                    print(f"   {key}: {value}")
            print()


# Example usage and tests
if __name__ == "__main__":
    print("Data Validation Tests")
    print("="*70)
    print()
    print("This module provides validators for:")
    print("  1. UnifiedDataValidator - validates commodity.silver.unified_data")
    print("  2. ForecastOutputValidator - validates forecast output tables")
    print()
    print("Usage:")
    print("  validator = UnifiedDataValidator(df)")
    print("  results = validator.validate_all()")
    print("  validator.print_report()")
