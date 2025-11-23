"""
Delete Locally-Trained Models from Database

Removes all models that were trained locally to fix version compatibility issues.
After running this, retrain models in Databricks to ensure NumPy/sklearn/xgboost version consistency.
"""

import os
import argparse
from databricks import sql

def delete_local_models(commodities=['Coffee', 'Sugar'], models=None, dry_run=True):
    """
    Delete locally-trained models from database.

    Args:
        commodities: List of commodities to clear
        models: List of model versions to clear (None = all models)
        dry_run: If True, only show what would be deleted
    """
    # Get credentials from environment
    host = os.environ['DATABRICKS_HOST'].replace('https://', '')
    http_path = os.environ['DATABRICKS_HTTP_PATH']
    token = os.environ['DATABRICKS_TOKEN']

    # Connect
    connection = sql.connect(
        server_hostname=host,
        http_path=http_path,
        access_token=token
    )

    cursor = connection.cursor()

    try:
        for commodity in commodities:
            print(f"\n{'='*80}")
            print(f"Commodity: {commodity}")
            print('='*80)

            # Build WHERE clause
            where_clauses = [f"commodity = '{commodity}'"]
            if models:
                models_str = "', '".join(models)
                where_clauses.append(f"model_version IN ('{models_str}')")

            where_clause = " AND ".join(where_clauses)

            # Count existing models
            count_query = f"""
            SELECT model_version, COUNT(*) as count
            FROM commodity.forecast.trained_models
            WHERE {where_clause}
            GROUP BY model_version
            ORDER BY model_version
            """

            cursor.execute(count_query)
            results = cursor.fetchall()

            if not results:
                print(f"  No models found for {commodity}")
                continue

            print(f"\nModels to {'DELETE' if not dry_run else 'DELETE (DRY RUN)'}:")
            total = 0
            for row in results:
                print(f"  {row[0]}: {row[1]} models")
                total += row[1]

            print(f"\nTotal: {total} models")

            if not dry_run:
                # Delete
                delete_query = f"""
                DELETE FROM commodity.forecast.trained_models
                WHERE {where_clause}
                """

                cursor.execute(delete_query)
                print(f"\n✅ Deleted {total} models for {commodity}")

                # Verify
                cursor.execute(count_query)
                verify = cursor.fetchall()
                if not verify:
                    print(f"✅ Verified: All models deleted for {commodity}")
                else:
                    print(f"⚠️  Warning: {len(verify)} model types still remain")

    finally:
        cursor.close()
        connection.close()

    print(f"\n{'='*80}")
    if dry_run:
        print("DRY RUN COMPLETE - No changes made")
        print("Run with --execute to actually delete models")
    else:
        print("✅ DELETION COMPLETE")
        print("\nNext steps:")
        print("1. Go to Databricks workspace")
        print("2. Run training script on existing cluster:")
        print("   python train_models.py --commodity Coffee --models naive xgboost sarimax_auto_weather \\")
        print("     --train-frequency semiannually --start-date 2018-01-01 --end-date 2025-11-17")
        print("3. Then run backfill in same environment")
    print('='*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Delete locally-trained models from database')
    parser.add_argument('--commodities', nargs='+', default=['Coffee', 'Sugar'],
                        help='Commodities to clear (default: Coffee Sugar)')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Model versions to clear (default: all)')
    parser.add_argument('--execute', action='store_true',
                        help='Actually delete (default is dry-run)')

    args = parser.parse_args()

    delete_local_models(
        commodities=args.commodities,
        models=args.models,
        dry_run=not args.execute
    )
