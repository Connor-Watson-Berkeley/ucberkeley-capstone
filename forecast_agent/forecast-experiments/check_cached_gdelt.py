"""
Check the cached GDELT data to diagnose why it's all NULLs
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("CHECKING CACHED GDELT DATA")
print("=" * 80)
print()

# Load the cached file
df = pd.read_parquet('data/unified_data_with_sentiment.parquet')

print(f"Total rows: {len(df):,}")
print(f"Total columns: {len(df.columns)}")
print()

# Identify GDELT columns
gdelt_cols = [c for c in df.columns if c.startswith('group_') or c.startswith('theme_')]
print(f"GDELT columns ({len(gdelt_cols)}): {gdelt_cols}")
print()

# Check NULL percentages
print("GDELT NULL percentages:")
for col in gdelt_cols:
    null_pct = df[col].isnull().mean() * 100
    print(f"  {col}: {null_pct:.1f}%")
print()

# Check if ANY row has GDELT data
has_gdelt = df[gdelt_cols].notna().any(axis=1).sum()
print(f"Rows with ANY GDELT data: {has_gdelt} / {len(df)}")
print()

# Sample some rows to see what we have
print("Sample of data (first 5 rows):")
sample_cols = ['date', 'commodity', 'region', 'close'] + gdelt_cols[:3]
print(df[sample_cols].head())
print()

# Check date range
print(f"Date range in file: {df['date'].min()} to {df['date'].max()}")
print(f"Unique dates: {df['date'].nunique():,}")
print(f"Unique commodities: {df['commodity'].unique()}")
print(f"Unique regions: {df['region'].nunique()}")
print()

print("=" * 80)
print("DIAGNOSIS")
print("=" * 80)
print()
print("The data was successfully downloaded, but the LEFT JOIN didn't find")
print("any matches in the gdelt_wide_fillforward table.")
print()
print("Possible causes:")
print("1. Column name mismatch in join (article_date vs date)")
print("2. Commodity name case mismatch ('Coffee' vs 'coffee')")
print("3. Date format mismatch")
print("4. gdelt_wide_fillforward table might be empty")
print()
print("To fix, you need to:")
print("1. Grant SELECT permissions again on gdelt_wide_fillforward")
print("2. Run the diagnostic script to identify the exact issue")
print("3. Update the join condition in download_data_with_sentiment.py")
print("4. Re-download the data")
