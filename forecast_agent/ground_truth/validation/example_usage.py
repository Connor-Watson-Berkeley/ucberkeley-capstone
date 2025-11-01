"""Example usage of InputDataValidator.

This script demonstrates how to validate input data before model training.
Run as part of your data pipeline to ensure data quality.

Usage:
    python -m ground_truth.validation.example_usage

    # Or import in your code:
    from ground_truth.validation import InputDataValidator
"""

from pyspark.sql import SparkSession
from ground_truth.core.data_loader import load_unified_data
from ground_truth.validation import InputDataValidator
from ground_truth.core.logger import get_logger

logger = get_logger(__name__)


def validate_coffee_data():
    """Example: Validate Coffee data from unified_data table."""

    print("\n" + "="*80)
    print("EXAMPLE: Validating Coffee Input Data")
    print("="*80 + "\n")

    # 1. Create Spark session
    spark = SparkSession.builder \
        .master("local[*]") \
        .appName("validate_coffee_data") \
        .getOrCreate()

    # 2. Load data
    logger.info("Loading Coffee data from unified_data table")
    df_spark = load_unified_data(
        spark,
        table_name="commodity.silver.unified_data",
        commodity='Coffee'
    )

    logger.info(f"Loaded {df_spark.count():,} rows")

    # 3. Create validator
    validator = InputDataValidator(df_spark, commodity='Coffee')

    # 4. Run all validation checks
    logger.info("Running validation checks...")
    results = validator.validate_all()

    # 5. Print report
    validator.print_report()

    # 6. Handle results
    if results['passed']:
        logger.info("✓ Data validation PASSED - Safe to train models")
        return True
    else:
        logger.error("✗ Data validation FAILED - Do not train models!")
        logger.error(f"Failed {results['summary']['failed_checks']} checks")

        # Log specific failures
        for check_name, check_result in results['checks'].items():
            if not check_result.get('passed'):
                logger.error(f"  - {check_name} failed")

        return False


def validate_and_alert():
    """Example: Validate data and send alert if fails (production pattern)."""

    from ground_truth.validation import InputDataValidator

    spark = SparkSession.builder.getOrCreate()
    df_spark = load_unified_data(spark, commodity='Coffee')

    validator = InputDataValidator(df_spark, commodity='Coffee')
    results = validator.validate_all()

    if not results['passed']:
        # In production, send alert (email, Slack, PagerDuty, etc.)
        logger.critical("DATA QUALITY ALERT: Validation failed!")

        # Example: Log to monitoring system
        validation_metrics = {
            'timestamp': results['validated_at'],
            'pass_rate': results['summary']['pass_rate'],
            'failed_checks': results['summary']['failed_checks']
        }

        logger.error(f"Validation metrics: {validation_metrics}")

        # Stop pipeline
        raise RuntimeError("Data validation failed - stopping pipeline")

    logger.info("Data validation passed - continuing pipeline")


def custom_validation_checks():
    """Example: Run specific validation checks (not all)."""

    spark = SparkSession.builder.getOrCreate()
    df_spark = load_unified_data(spark, commodity='Coffee')

    validator = InputDataValidator(df_spark, commodity='Coffee')

    # Run only specific checks
    schema_check = validator.validate_schema()
    if not schema_check['passed']:
        logger.error("Schema validation failed!")
        logger.error(f"Missing columns: {schema_check['missing_columns']}")

    null_check = validator.validate_no_nulls_in_critical_fields()
    if not null_check['passed']:
        logger.error("Null check failed!")
        logger.error(f"Fields with nulls: {null_check['failed_fields']}")

    freshness_check = validator.validate_data_freshness(max_days_old=3)
    if not freshness_check['passed']:
        logger.warning(f"Data is {freshness_check['days_old']} days old")


if __name__ == "__main__":
    import sys

    # Run validation
    success = validate_coffee_data()

    # Exit with appropriate code
    sys.exit(0 if success else 1)
