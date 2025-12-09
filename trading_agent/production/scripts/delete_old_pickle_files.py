"""
Delete old pickle files to ensure fresh backtest run
"""
from pathlib import Path

print("=" * 100)
print("DELETING OLD BACKTEST PICKLE FILES")
print("=" * 100)

volume_path = Path("/Volumes/commodity/trading_agent/files/")

# Find all pickle files
pickle_files = list(volume_path.glob("results_detailed_*.pkl"))
print(f"\nFound {len(pickle_files)} pickle files to delete")

# Delete each file
deleted_count = 0
for pickle_file in pickle_files:
    try:
        size_mb = pickle_file.stat().st_size / (1024 * 1024)
        pickle_file.unlink()
        print(f"  ✓ Deleted: {pickle_file.name} ({size_mb:.1f} MB)")
        deleted_count += 1
    except Exception as e:
        print(f"  ✗ Failed to delete {pickle_file.name}: {e}")

print(f"\n{'=' * 100}")
print(f"DELETION COMPLETE: {deleted_count}/{len(pickle_files)} files deleted")
print(f"{'=' * 100}")
