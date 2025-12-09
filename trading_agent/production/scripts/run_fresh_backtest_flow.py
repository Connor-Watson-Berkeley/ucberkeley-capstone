"""
Run complete fresh backtest flow:
1. Delete old pickle files
2. Generate forecast manifests
3. Run backtests for validated models
4. Save results

This ensures all results are fresh and consistent with current forecast data.
"""
import sys
import os

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = '/Workspace/Repos/Project_Git/ucberkeley-capstone/trading_agent/production/scripts'

# Add trading_agent to path
trading_agent_dir = os.path.dirname(os.path.dirname(script_dir))
if trading_agent_dir not in sys.path:
    sys.path.insert(0, trading_agent_dir)

from pathlib import Path
from pyspark.sql import SparkSession
from production.config import COMMODITY_CONFIGS, VOLUME_PATH
from production.runners.multi_commodity_runner import MultiCommodityRunner

def main():
    spark = SparkSession.builder.getOrCreate()

    print("\n" + "=" * 100)
    print("FRESH BACKTEST FLOW - COMPLETE PIPELINE")
    print("=" * 100)

    # Step 1: Delete old pickle files
    print("\n" + "=" * 100)
    print("STEP 1: DELETING OLD PICKLE FILES")
    print("=" * 100)

    volume_path = Path(VOLUME_PATH)
    pickle_files = list(volume_path.glob("results_detailed_*.pkl"))
    print(f"\nFound {len(pickle_files)} pickle files to delete")

    deleted_count = 0
    for pickle_file in pickle_files:
        try:
            size_mb = pickle_file.stat().st_size / (1024 * 1024)
            pickle_file.unlink()
            print(f"  ✓ Deleted: {pickle_file.name} ({size_mb:.1f} MB)")
            deleted_count += 1
        except Exception as e:
            print(f"  ✗ Failed: {pickle_file.name}: {e}")

    print(f"\n✓ Deletion complete: {deleted_count}/{len(pickle_files)} files deleted")

    # Step 2: Generate forecast manifests
    print("\n" + "=" * 100)
    print("STEP 2: GENERATING FORECAST MANIFESTS")
    print("=" * 100)

    import json
    from datetime import datetime

    for commodity in COMMODITY_CONFIGS.keys():
        print(f"\nGenerating manifest for {commodity.upper()}...")

        # Get all model versions for this commodity
        commodity_cap = commodity.capitalize()
        models_df = spark.sql(f"""
            SELECT DISTINCT model_version
            FROM commodity.forecast.distributions
            WHERE commodity = '{commodity_cap}'
                AND is_actuals = false
            ORDER BY model_version
        """)

        model_versions = [row.model_version for row in models_df.collect()]
        print(f"  Found {len(model_versions)} models: {model_versions}")

        manifest = {
            'commodity': commodity,
            'generated_at': datetime.now().isoformat(),
            'models': {}
        }

        for model_version in model_versions:
            # Get forecast date coverage
            coverage_df = spark.sql(f"""
                SELECT
                    MIN(forecast_start_date) as first_pred,
                    MAX(forecast_start_date) as last_pred,
                    COUNT(DISTINCT forecast_start_date) as n_dates,
                    COUNT(DISTINCT YEAR(forecast_start_date)) as n_years
                FROM commodity.forecast.distributions
                WHERE commodity = '{commodity_cap}'
                    AND model_version = '{model_version}'
                    AND is_actuals = false
            """)

            row = coverage_df.first()

            if row and row.n_dates > 0:
                first_date = datetime.strptime(str(row.first_pred), '%Y-%m-%d')
                last_date = datetime.strptime(str(row.last_pred), '%Y-%m-%d')
                years_span = (last_date - first_date).days / 365.25
                expected_days = (last_date - first_date).days + 1
                coverage_pct = (row.n_dates / expected_days) * 100

                # Determine quality
                if coverage_pct >= 90 and years_span >= 5:
                    quality = 'EXCELLENT'
                    meets_criteria = True
                elif coverage_pct >= 70 and years_span >= 3:
                    quality = 'GOOD'
                    meets_criteria = True
                elif coverage_pct >= 50:
                    quality = 'MARGINAL'
                    meets_criteria = False
                else:
                    quality = 'SPARSE'
                    meets_criteria = False

                manifest['models'][model_version] = {
                    'type': 'synthetic' if 'synthetic' in model_version else 'real',
                    'date_range': {
                        'start': str(row.first_pred),
                        'end': str(row.last_pred)
                    },
                    'years_span': round(years_span, 2),
                    'expected_days': expected_days,
                    'prediction_dates': row.n_dates,
                    'coverage_pct': round(coverage_pct, 2),
                    'years_available': row.n_years,
                    'meets_criteria': meets_criteria,
                    'quality': quality,
                    'pickle_file': f'prediction_matrices_{commodity.lower()}_{model_version}.pkl'
                }

                print(f"    ✓ {model_version}: {row.n_dates} dates, {round(coverage_pct, 1)}% coverage, {quality}")

        # Write manifest
        manifest_path = volume_path / f'forecast_manifest_{commodity}.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        print(f"  ✓ Manifest created: {manifest_path}")

    # Step 3: Run backtests
    print("\n" + "=" * 100)
    print("STEP 3: RUNNING BACKTESTS FOR ALL VALIDATED MODELS")
    print("=" * 100)

    runner = MultiCommodityRunner(
        spark=spark,
        commodity_configs=COMMODITY_CONFIGS,
        volume_path=str(volume_path),
        output_schema="commodity.trading_agent",
        use_optimized_params=False,
        run_statistical_tests=False  # We'll run stats separately
    )

    # Run all commodities and all models (will auto-discover from manifests)
    results = runner.run_all_commodities(
        commodities=list(COMMODITY_CONFIGS.keys()),
        verbose=True
    )

    print("\n" + "=" * 100)
    print("FRESH BACKTEST FLOW COMPLETE")
    print("=" * 100)
    print("\nResults saved:")
    print("  • Pickle files: {volume_path}/results_detailed_{{commodity}}_{{model}}.pkl")
    print("  • Delta tables: commodity.trading_agent.results_{{commodity}}_by_year_{{model}}")
    print("\nReady for statistical analysis!")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
