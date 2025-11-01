"""Input data validation for unified_data table.

Validates data quality before model training:
- Schema compliance
- Null detection in critical fields
- Anomalous values (outliers, impossible values)
- Data freshness
- Completeness (row counts, regional coverage)
"""

from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from typing import Dict, List
from datetime import datetime, timedelta
from ground_truth.core.logger import get_logger

logger = get_logger(__name__)


class InputDataValidator:
    """Validates unified_data table before model training.

    Usage:
        from ground_truth.validation import InputDataValidator

        validator = InputDataValidator(df_spark)
        results = validator.validate_all()

        if not results['passed']:
            logger.error("Data validation failed!")
            validator.print_report()
    """

    # Expected schema for commodity.silver.unified_data
    EXPECTED_SCHEMA = {
        'date': 'date',
        'commodity': 'string',
        'region': 'string',
        'close': 'double',
        'temp_c': 'double',
        'humidity_pct': 'double',
        'precipitation_mm': 'double',
        'vix': 'double'
    }

    # Critical fields that should never be null
    CRITICAL_FIELDS = ['date', 'commodity', 'region', 'close']

    # Acceptable null rates for weather fields (can have some nulls if sparse data)
    MAX_NULL_RATE = {
        'temp_c': 0.05,         # Max 5% nulls
        'humidity_pct': 0.05,
        'precipitation_mm': 0.10,  # Precipitation can be more sparse
        'vix': 0.02,            # Market data should be complete
    }

    # Value ranges for anomaly detection
    VALUE_RANGES = {
        'close': (0, 1000),           # Coffee/Sugar prices
        'temp_c': (-50, 60),          # Temperature in Celsius
        'humidity_pct': (0, 100),     # Percentage
        'precipitation_mm': (0, 500), # Daily precipitation
        'vix': (0, 100),              # VIX typically 10-80
    }

    def __init__(self, df_spark: SparkDataFrame, commodity: str = None):
        """Initialize validator.

        Args:
            df_spark: Spark DataFrame to validate
            commodity: Optional commodity filter ('Coffee', 'Sugar')
        """
        self.df_spark = df_spark
        self.commodity = commodity
        self.validation_results = {}

        if commodity:
            self.df_spark = self.df_spark.filter(F.col('commodity') == commodity)
            logger.info(f"Validating data for commodity: {commodity}")

    def validate_all(self) -> Dict:
        """Run all validation checks.

        Returns:
            Dict with:
                - passed: bool (True if all checks passed)
                - checks: Dict of individual check results
                - summary: Dict with overall statistics
        """
        logger.info("Starting comprehensive data validation")

        checks = {}
        checks['schema'] = self.validate_schema()
        checks['nulls_critical'] = self.validate_no_nulls_in_critical_fields()
        checks['null_rates'] = self.validate_acceptable_null_rates()
        checks['value_ranges'] = self.validate_value_ranges()
        checks['freshness'] = self.validate_data_freshness()
        checks['completeness'] = self.validate_completeness()
        checks['duplicates'] = self.validate_no_duplicates()

        # Overall pass/fail
        all_passed = all(check.get('passed', False) for check in checks.values())

        result = {
            'passed': all_passed,
            'checks': checks,
            'summary': self._generate_summary(checks),
            'validated_at': datetime.now().isoformat()
        }

        self.validation_results = result
        return result

    def validate_schema(self) -> Dict:
        """Check that all expected columns exist with correct types."""
        logger.debug("Validating schema")

        actual_schema = {field.name: str(field.dataType) for field in self.df_spark.schema.fields}
        missing_columns = []
        type_mismatches = []

        for col_name, expected_type in self.EXPECTED_SCHEMA.items():
            if col_name not in actual_schema:
                missing_columns.append(col_name)
            elif expected_type not in actual_schema[col_name].lower():
                type_mismatches.append({
                    'column': col_name,
                    'expected': expected_type,
                    'actual': actual_schema[col_name]
                })

        passed = len(missing_columns) == 0 and len(type_mismatches) == 0

        return {
            'passed': passed,
            'missing_columns': missing_columns,
            'type_mismatches': type_mismatches,
            'actual_columns': list(actual_schema.keys())
        }

    def validate_no_nulls_in_critical_fields(self) -> Dict:
        """Check that critical fields have no nulls."""
        logger.debug("Checking nulls in critical fields")

        null_counts = {}
        total_rows = self.df_spark.count()

        for field in self.CRITICAL_FIELDS:
            if field in [f.name for f in self.df_spark.schema.fields]:
                null_count = self.df_spark.filter(F.col(field).isNull()).count()
                null_counts[field] = {
                    'count': null_count,
                    'percentage': (null_count / total_rows * 100) if total_rows > 0 else 0
                }

        failed_fields = {k: v for k, v in null_counts.items() if v['count'] > 0}
        passed = len(failed_fields) == 0

        return {
            'passed': passed,
            'null_counts': null_counts,
            'failed_fields': list(failed_fields.keys()),
            'total_rows': total_rows
        }

    def validate_acceptable_null_rates(self) -> Dict:
        """Check that non-critical fields have acceptable null rates."""
        logger.debug("Checking null rates for non-critical fields")

        null_rates = {}
        total_rows = self.df_spark.count()
        violations = []

        for field, max_rate in self.MAX_NULL_RATE.items():
            if field in [f.name for f in self.df_spark.schema.fields]:
                null_count = self.df_spark.filter(F.col(field).isNull()).count()
                actual_rate = (null_count / total_rows) if total_rows > 0 else 0

                null_rates[field] = {
                    'null_count': null_count,
                    'null_rate': actual_rate,
                    'max_allowed': max_rate,
                    'passed': actual_rate <= max_rate
                }

                if actual_rate > max_rate:
                    violations.append({
                        'field': field,
                        'actual_rate': actual_rate,
                        'max_allowed': max_rate
                    })

        passed = len(violations) == 0

        return {
            'passed': passed,
            'null_rates': null_rates,
            'violations': violations
        }

    def validate_value_ranges(self) -> Dict:
        """Check for anomalous values outside expected ranges."""
        logger.debug("Validating value ranges")

        anomalies = []
        total_rows = self.df_spark.count()

        for field, (min_val, max_val) in self.VALUE_RANGES.items():
            if field in [f.name for f in self.df_spark.schema.fields]:
                # Count values outside range
                out_of_range = self.df_spark.filter(
                    (F.col(field) < min_val) | (F.col(field) > max_val)
                ).count()

                if out_of_range > 0:
                    # Get example outliers
                    examples = self.df_spark.filter(
                        (F.col(field) < min_val) | (F.col(field) > max_val)
                    ).select('date', 'region', field).limit(5).collect()

                    anomalies.append({
                        'field': field,
                        'count': out_of_range,
                        'percentage': (out_of_range / total_rows * 100) if total_rows > 0 else 0,
                        'expected_range': (min_val, max_val),
                        'examples': [
                            {
                                'date': str(row['date']),
                                'region': row['region'],
                                'value': float(row[field])
                            } for row in examples
                        ]
                    })

        passed = len(anomalies) == 0

        return {
            'passed': passed,
            'anomalies': anomalies,
            'fields_checked': list(self.VALUE_RANGES.keys())
        }

    def validate_data_freshness(self, max_days_old: int = 7) -> Dict:
        """Check that data is recent (within last N days)."""
        logger.debug("Checking data freshness")

        max_date = self.df_spark.agg(F.max('date')).collect()[0][0]
        days_old = (datetime.now().date() - max_date).days if max_date else None

        passed = days_old is not None and days_old <= max_days_old

        return {
            'passed': passed,
            'latest_date': str(max_date) if max_date else None,
            'days_old': days_old,
            'max_allowed_days': max_days_old
        }

    def validate_completeness(self) -> Dict:
        """Check data completeness (row counts, regional coverage)."""
        logger.debug("Checking data completeness")

        total_rows = self.df_spark.count()
        unique_dates = self.df_spark.select('date').distinct().count()
        unique_regions = self.df_spark.select('region').distinct().count()

        # Get date range
        date_stats = self.df_spark.agg(
            F.min('date').alias('min_date'),
            F.max('date').alias('max_date')
        ).collect()[0]

        min_date = date_stats['min_date']
        max_date = date_stats['max_date']
        expected_days = (max_date - min_date).days + 1 if min_date and max_date else 0

        # Calculate expected vs actual rows
        expected_rows = expected_days * unique_regions
        completeness_rate = (total_rows / expected_rows) if expected_rows > 0 else 0

        # Thresholds
        min_rows = 10000  # Minimum reasonable dataset size
        min_regions = 5   # Minimum regional coverage
        min_completeness = 0.80  # At least 80% complete (accounts for weekends, holidays)

        passed = (
            total_rows >= min_rows and
            unique_regions >= min_regions and
            completeness_rate >= min_completeness
        )

        return {
            'passed': passed,
            'total_rows': total_rows,
            'unique_dates': unique_dates,
            'unique_regions': unique_regions,
            'date_range': {
                'min': str(min_date) if min_date else None,
                'max': str(max_date) if max_date else None,
                'days': expected_days
            },
            'completeness_rate': completeness_rate,
            'thresholds': {
                'min_rows': min_rows,
                'min_regions': min_regions,
                'min_completeness': min_completeness
            }
        }

    def validate_no_duplicates(self) -> Dict:
        """Check for duplicate rows (same date, commodity, region)."""
        logger.debug("Checking for duplicates")

        total_rows = self.df_spark.count()
        distinct_rows = self.df_spark.select('date', 'commodity', 'region').distinct().count()

        duplicate_count = total_rows - distinct_rows
        passed = duplicate_count == 0

        # If duplicates found, get examples
        examples = []
        if duplicate_count > 0:
            duplicates_df = self.df_spark.groupBy('date', 'commodity', 'region').count().filter(F.col('count') > 1)
            examples = duplicates_df.limit(5).collect()

        return {
            'passed': passed,
            'total_rows': total_rows,
            'distinct_rows': distinct_rows,
            'duplicate_count': duplicate_count,
            'examples': [
                {
                    'date': str(row['date']),
                    'commodity': row['commodity'],
                    'region': row['region'],
                    'count': int(row['count'])
                } for row in examples
            ] if examples else []
        }

    def _generate_summary(self, checks: Dict) -> Dict:
        """Generate overall summary of validation results."""
        total_checks = len(checks)
        passed_checks = sum(1 for check in checks.values() if check.get('passed', False))

        return {
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'failed_checks': total_checks - passed_checks,
            'pass_rate': (passed_checks / total_checks * 100) if total_checks > 0 else 0
        }

    def print_report(self):
        """Print human-readable validation report."""
        if not self.validation_results:
            logger.warning("No validation results available. Run validate_all() first.")
            return

        print("\n" + "="*80)
        print("INPUT DATA VALIDATION REPORT")
        print("="*80)

        summary = self.validation_results['summary']
        print(f"\nOverall: {summary['passed_checks']}/{summary['total_checks']} checks passed "
              f"({summary['pass_rate']:.1f}%)")

        if self.validation_results['passed']:
            print("✓ All validation checks PASSED")
        else:
            print("✗ Some validation checks FAILED")

        print("\n" + "-"*80)
        print("DETAILED RESULTS")
        print("-"*80)

        for check_name, result in self.validation_results['checks'].items():
            status = "✓ PASS" if result.get('passed') else "✗ FAIL"
            print(f"\n{check_name.upper()}: {status}")

            # Print specific details for each check type
            if check_name == 'schema' and not result['passed']:
                if result['missing_columns']:
                    print(f"  Missing columns: {result['missing_columns']}")
                if result['type_mismatches']:
                    for mismatch in result['type_mismatches']:
                        print(f"  Type mismatch: {mismatch['column']} "
                              f"(expected {mismatch['expected']}, got {mismatch['actual']})")

            elif check_name == 'nulls_critical' and not result['passed']:
                for field in result['failed_fields']:
                    counts = result['null_counts'][field]
                    print(f"  {field}: {counts['count']} nulls ({counts['percentage']:.2f}%)")

            elif check_name == 'null_rates' and not result['passed']:
                for violation in result['violations']:
                    print(f"  {violation['field']}: {violation['actual_rate']*100:.2f}% nulls "
                          f"(max allowed: {violation['max_allowed']*100:.1f}%)")

            elif check_name == 'value_ranges' and not result['passed']:
                for anomaly in result['anomalies']:
                    print(f"  {anomaly['field']}: {anomaly['count']} outliers "
                          f"({anomaly['percentage']:.2f}%) outside {anomaly['expected_range']}")
                    if anomaly['examples']:
                        print(f"    Examples: {anomaly['examples'][:3]}")

            elif check_name == 'freshness':
                print(f"  Latest data: {result['latest_date']} ({result['days_old']} days old)")

            elif check_name == 'completeness':
                print(f"  Total rows: {result['total_rows']:,}")
                print(f"  Regions: {result['unique_regions']}")
                print(f"  Date range: {result['date_range']['min']} to {result['date_range']['max']}")
                print(f"  Completeness: {result['completeness_rate']*100:.1f}%")

            elif check_name == 'duplicates' and not result['passed']:
                print(f"  Duplicate rows found: {result['duplicate_count']}")

        print("\n" + "="*80)
        print(f"Validated at: {self.validation_results['validated_at']}")
        print("="*80 + "\n")
