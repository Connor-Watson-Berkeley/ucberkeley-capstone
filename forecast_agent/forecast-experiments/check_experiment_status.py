"""Quick status check for all running experiments"""
import pandas as pd
import os

print("=" * 80)
print("EXPERIMENT STATUS")
print("=" * 80)
print()

# Comprehensive experiments
if os.path.exists('experiment_results_comprehensive.csv'):
    df_comp = pd.read_csv('experiment_results_comprehensive.csv')
    df_comp_success = df_comp[df_comp['success']]
    print("1. COMPREHENSIVE DARTS EXPERIMENTS")
    print(f"   Progress: {len(df_comp)}/128 ({len(df_comp)/128*100:.1f}%)")
    if len(df_comp_success) > 0:
        best_idx = df_comp_success['mape'].argmin()
        print(f"   Best: {df_comp_success.iloc[best_idx]['model']} @ {df_comp_success.iloc[best_idx]['region']}, {df_comp_success.iloc[best_idx]['feature_set']}, {int(df_comp_success.iloc[best_idx]['horizon_days'])}-day = {df_comp_success.iloc[best_idx]['mape']:.2f}% MAPE")
        print(f"   Latest: {df_comp.iloc[-1]['model']} @ {df_comp.iloc[-1]['region']}, {df_comp.iloc[-1]['feature_set']}, {int(df_comp.iloc[-1]['horizon_days'])}-day")
    print()

# Pivot experiments
if os.path.exists('experiment_results_pivoted_regional.csv'):
    df_pivot = pd.read_csv('experiment_results_pivoted_regional.csv')
    df_pivot_success = df_pivot[df_pivot['success']]
    print("2. MULTI-REGIONAL PIVOT EXPERIMENTS (176 features)")
    print(f"   Progress: {len(df_pivot)}/64 ({len(df_pivot)/64*100:.1f}%)")
    if len(df_pivot_success) > 0:
        best_idx = df_pivot_success['mape'].argmin()
        print(f"   Best: {df_pivot_success.iloc[best_idx]['model']} @ {int(df_pivot_success.iloc[best_idx]['horizon_days'])}-day = {df_pivot_success.iloc[best_idx]['mape']:.2f}% MAPE")
        print(f"   Latest: {df_pivot.iloc[-1]['model']} @ {int(df_pivot.iloc[-1]['horizon_days'])}-day = {df_pivot.iloc[-1]['mape']:.2f}% MAPE")
    print()

# Tree/linear experiments
if os.path.exists('experiment_results_tree_linear_pivot.csv'):
    df_tree = pd.read_csv('experiment_results_tree_linear_pivot.csv')
    df_tree_success = df_tree[df_tree['success']]
    print("3. TREE & LINEAR MODELS (lagged features)")
    print(f"   Progress: {len(df_tree)}/20 ({len(df_tree)/20*100:.1f}%)")
    if len(df_tree_success) > 0:
        best_idx = df_tree_success['mape'].argmin()
        print(f"   Best: {df_tree_success.iloc[best_idx]['model']} @ {int(df_tree_success.iloc[best_idx]['horizon_days'])}-day = {df_tree_success.iloc[best_idx]['mape']:.2f}% MAPE")
        print(f"   Latest: {df_tree.iloc[-1]['model']} @ {int(df_tree.iloc[-1]['horizon_days'])}-day")
    print()
else:
    print("3. TREE & LINEAR MODELS")
    print("   Status: Running (no results yet)")
    print()

print("=" * 80)
