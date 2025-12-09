"""
Archive old backtest results to avoid confusion with new runs
Moves pickle files and documents what was archived
"""
import os
import shutil
from pathlib import Path
from datetime import datetime

print("=" * 100)
print("ARCHIVING OLD BACKTEST RESULTS")
print("=" * 100)

# Paths
volume_path = Path("/Volumes/commodity/trading_agent/files/")
archive_path = volume_path / f"archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Create archive directory
archive_path.mkdir(exist_ok=True)
print(f"\n✓ Created archive directory: {archive_path}")

# Find all pickle files
pickle_files = list(volume_path.glob("results_detailed_*.pkl"))
print(f"\n✓ Found {len(pickle_files)} pickle files to archive")

# Archive each file
archived_count = 0
for pickle_file in pickle_files:
    try:
        dest = archive_path / pickle_file.name
        shutil.move(str(pickle_file), str(dest))
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  ✓ Archived: {pickle_file.name} ({size_mb:.1f} MB)")
        archived_count += 1
    except Exception as e:
        print(f"  ✗ Failed to archive {pickle_file.name}: {e}")

# Create archive manifest
manifest_file = archive_path / "ARCHIVE_MANIFEST.txt"
with open(manifest_file, 'w') as f:
    f.write(f"BACKTEST RESULTS ARCHIVE\n")
    f.write(f"{'=' * 80}\n\n")
    f.write(f"Archived at: {datetime.now().isoformat()}\n")
    f.write(f"Reason: Preparing for fresh backtest runs with validated forecast data\n\n")
    f.write(f"Files archived: {archived_count}\n\n")
    f.write(f"Archived files:\n")
    for pickle_file in sorted(archive_path.glob("*.pkl")):
        size_mb = pickle_file.stat().st_size / (1024 * 1024)
        f.write(f"  - {pickle_file.name} ({size_mb:.1f} MB)\n")

print(f"\n✓ Created archive manifest: {manifest_file}")

print("\n" + "=" * 100)
print(f"ARCHIVE COMPLETE: {archived_count} files moved to {archive_path.name}")
print("=" * 100)
print("\nNext steps:")
print("1. Run backtests for the 4 validated models:")
print("   - coffee/naive (EXCELLENT)")
print("   - coffee/xgboost (EXCELLENT)")
print("   - coffee/sarimax_auto_weather (MARGINAL)")
print("   - sugar/naive (MARGINAL)")
print("2. Run statistical tests on fresh results")
