import os
from pathlib import Path

print("=" * 100)
print("CHECKING FOR BACKTEST PICKLE FILES")
print("=" * 100)

# Check Databricks volume path
volume_path = Path("/dbfs/mnt/commodity/trading_agent/files/")
if volume_path.exists():
    print(f"\n✓ Volume path exists: {volume_path}")

    # List all pickle files
    pickle_files = list(volume_path.glob("results_detailed_*.pkl"))
    print(f"\nFound {len(pickle_files)} pickle files:")
    for f in sorted(pickle_files):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  • {f.name} ({size_mb:.1f} MB)")

    # Check for specific models
    print("\n" + "=" * 100)
    print("CHECKING FOR MANIFEST-IDENTIFIED MODELS")
    print("=" * 100)

    models_to_check = [
        ('coffee', 'naive'),
        ('coffee', 'xgboost'),
        ('coffee', 'sarimax_auto_weather'),
        ('sugar', 'naive'),
    ]

    for commodity, model in models_to_check:
        filename = f"results_detailed_{commodity}_{model}.pkl"
        filepath = volume_path / filename

        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  ✓ {commodity}/{model}: {size_mb:.1f} MB")
        else:
            print(f"  ✗ {commodity}/{model}: NO PICKLE FILE")
else:
    print(f"\n✗ Volume path does not exist: {volume_path}")

    # Try alternative path
    alt_path = Path("/Volumes/commodity/trading_agent/files/")
    if alt_path.exists():
        print(f"✓ Alternative path exists: {alt_path}")
        pickle_files = list(alt_path.glob("results_detailed_*.pkl"))
        print(f"Found {len(pickle_files)} pickle files")
        for f in sorted(pickle_files)[:10]:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  • {f.name} ({size_mb:.1f} MB)")
    else:
        print(f"✗ Alternative path also missing: {alt_path}")

print("\n" + "=" * 100)
