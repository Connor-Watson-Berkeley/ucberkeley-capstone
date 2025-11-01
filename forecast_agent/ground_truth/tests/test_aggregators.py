"""Unit tests for regional aggregation strategies.

Tests:
- aggregate_regions_mean: Simple averaging across regions
- aggregate_regions_weighted: Production-weighted averaging
- pivot_regions_as_features: Each region as separate column
"""

import pytest
from ground_truth.features.aggregators import (
    aggregate_regions_mean,
    aggregate_regions_weighted,
    pivot_regions_as_features
)


class TestAggregateRegionsMean:
    """Test simple mean aggregation."""

    def test_basic_mean_aggregation(self, spark, sample_unified_data):
        """Test that mean aggregation averages correctly."""
        # Aggregate Coffee data
        result = aggregate_regions_mean(
            sample_unified_data,
            commodity='Coffee',
            features=['close', 'temp_c', 'humidity_pct', 'precipitation_mm']
        )

        result_pd = result.toPandas()

        # Check: Should have 10 dates (one per day)
        assert len(result_pd) == 10

        # Check: Close should be same (185.5 for day 1) for all regions
        first_row = result_pd[result_pd['date'] == '2024-01-01'].iloc[0]
        assert first_row['close'] == 185.5

        # Check: Temp should be average of Colombia (25.1) and Vietnam (30.1) = 27.6
        assert pytest.approx(first_row['temp_c'], rel=0.01) == 27.6

        # Check: Humidity should be average of 75.5 and 80.5 = 78.0
        assert pytest.approx(first_row['humidity_pct'], rel=0.01) == 78.0

    def test_commodity_filtering(self, spark, sample_unified_data):
        """Test that commodity filter works correctly."""
        # Get Coffee data
        coffee_result = aggregate_regions_mean(
            sample_unified_data,
            commodity='Coffee',
            features=['close', 'temp_c']
        ).toPandas()

        # Get Sugar data
        sugar_result = aggregate_regions_mean(
            sample_unified_data,
            commodity='Sugar',
            features=['close', 'temp_c']
        ).toPandas()

        # Check: Both should have 10 dates
        assert len(coffee_result) == 10
        assert len(sugar_result) == 10

        # Check: Prices should be different (Coffee ~185, Sugar ~22)
        assert coffee_result['close'].iloc[0] > 100
        assert sugar_result['close'].iloc[0] < 50

    def test_cutoff_date_filtering(self, spark, sample_unified_data):
        """Test that cutoff_date filters correctly."""
        result = aggregate_regions_mean(
            sample_unified_data,
            commodity='Coffee',
            features=['close', 'temp_c'],
            cutoff_date='2024-01-05'
        ).toPandas()

        # Check: Should only have 5 dates (Jan 1-5)
        assert len(result) == 5

        # Check: Latest date should be Jan 5
        assert result['date'].max() == pd.Timestamp('2024-01-05')

    def test_missing_commodity(self, spark, sample_unified_data):
        """Test behavior with non-existent commodity."""
        result = aggregate_regions_mean(
            sample_unified_data,
            commodity='Cocoa',  # Doesn't exist
            features=['close', 'temp_c']
        ).toPandas()

        # Check: Should return empty DataFrame
        assert len(result) == 0


class TestAggregateRegionsWeighted:
    """Test production-weighted aggregation."""

    def test_weighted_aggregation(self, spark, sample_unified_data):
        """Test that weighted aggregation applies production weights correctly."""
        # Define weights: Colombia gets 60%, Vietnam gets 40%
        weights = {
            'Colombia': 0.6,
            'Vietnam': 0.4
        }

        result = aggregate_regions_weighted(
            sample_unified_data,
            commodity='Coffee',
            features=['close', 'temp_c'],
            production_weights=weights
        ).toPandas()

        # Check: Should have 10 dates
        assert len(result) == 10

        # Check: Weighted temp for day 1
        # Colombia: 25.1, Vietnam: 30.1
        # Weighted: 0.6*25.1 + 0.4*30.1 = 15.06 + 12.04 = 27.1
        first_row = result[result['date'] == '2024-01-01'].iloc[0]
        assert pytest.approx(first_row['temp_c'], rel=0.01) == 27.1

    def test_fallback_to_mean_without_weights(self, spark, sample_unified_data):
        """Test that function falls back to mean when no weights provided."""
        result_weighted = aggregate_regions_weighted(
            sample_unified_data,
            commodity='Coffee',
            features=['close', 'temp_c'],
            production_weights=None  # No weights
        ).toPandas()

        result_mean = aggregate_regions_mean(
            sample_unified_data,
            commodity='Coffee',
            features=['close', 'temp_c']
        ).toPandas()

        # Check: Results should be identical
        assert result_weighted['temp_c'].tolist() == result_mean['temp_c'].tolist()

    def test_weights_dont_sum_to_one(self, spark, sample_unified_data):
        """Test behavior when weights don't sum to 1.0."""
        # Weights sum to 0.8 (not 1.0)
        weights = {
            'Colombia': 0.5,
            'Vietnam': 0.3
        }

        result = aggregate_regions_weighted(
            sample_unified_data,
            commodity='Coffee',
            features=['close', 'temp_c'],
            production_weights=weights
        ).toPandas()

        # Check: Should still compute correctly (normalized by sum of weights)
        assert len(result) == 10
        # Weighted average: (0.5*25.1 + 0.3*30.1) / (0.5 + 0.3) = 26.625
        first_row = result[result['date'] == '2024-01-01'].iloc[0]
        assert pytest.approx(first_row['temp_c'], rel=0.01) == 26.625


class TestPivotRegionsAsFeatures:
    """Test pivoting regions into separate columns."""

    def test_pivot_creates_region_columns(self, spark, sample_unified_data):
        """Test that pivot creates one column per region per feature."""
        result = pivot_regions_as_features(
            sample_unified_data,
            commodity='Coffee',
            features=['close', 'temp_c', 'humidity_pct']
        ).toPandas()

        # Check: Should have 10 dates
        assert len(result) == 10

        # Check: Should have columns for each region
        expected_cols = [
            'date', 'commodity', 'close',  # Non-regional
            'temp_c_colombia', 'temp_c_vietnam',  # Regional temps
            'humidity_pct_colombia', 'humidity_pct_vietnam'  # Regional humidity
        ]

        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_pivot_values_correct(self, spark, sample_unified_data):
        """Test that pivoted values match original data."""
        result = pivot_regions_as_features(
            sample_unified_data,
            commodity='Coffee',
            features=['close', 'temp_c']
        ).toPandas()

        first_row = result[result['date'] == '2024-01-01'].iloc[0]

        # Check: Colombia temp should be 25.1
        assert pytest.approx(first_row['temp_c_colombia'], rel=0.01) == 25.1

        # Check: Vietnam temp should be 30.1
        assert pytest.approx(first_row['temp_c_vietnam'], rel=0.01) == 30.1

    def test_pivot_non_regional_features_not_duplicated(self, spark, sample_unified_data):
        """Test that non-regional features (close, vix) are not pivoted."""
        result = pivot_regions_as_features(
            sample_unified_data,
            commodity='Coffee',
            features=['close', 'temp_c']
        ).toPandas()

        # Check: Should NOT have close_colombia or close_vietnam
        assert 'close_colombia' not in result.columns
        assert 'close_vietnam' not in result.columns

        # Check: Should have single 'close' column
        assert 'close' in result.columns

    def test_pivot_many_regions(self, spark, sample_unified_data):
        """Test that pivot handles many regions correctly."""
        result = pivot_regions_as_features(
            sample_unified_data,
            commodity='Coffee',
            features=['temp_c', 'humidity_pct', 'precipitation_mm']
        ).toPandas()

        # Check: Should have 2 regions × 3 features = 6 regional columns
        regional_cols = [col for col in result.columns if 'colombia' in col or 'vietnam' in col]
        assert len(regional_cols) == 6


# Integration test
def test_all_aggregation_methods_return_same_row_count(spark, sample_unified_data):
    """Integration test: All aggregation methods should return same number of rows."""
    features = ['close', 'temp_c', 'humidity_pct']

    result_mean = aggregate_regions_mean(
        sample_unified_data, 'Coffee', features
    ).toPandas()

    result_weighted = aggregate_regions_weighted(
        sample_unified_data, 'Coffee', features,
        production_weights={'Colombia': 0.6, 'Vietnam': 0.4}
    ).toPandas()

    result_pivot = pivot_regions_as_features(
        sample_unified_data, 'Coffee', features
    ).toPandas()

    # Check: All should have 10 rows (10 dates)
    assert len(result_mean) == 10
    assert len(result_weighted) == 10
    assert len(result_pivot) == 10


# Import pandas for timestamp comparison
import pandas as pd
