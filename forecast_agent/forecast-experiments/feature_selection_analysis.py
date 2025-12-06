"""
Feature Selection Analysis for Multi-Regional Weather Forecasting

Identifies the most important features from the 176+ dimensional pivoted weather data
using multiple techniques and ensemble voting.

Run this AFTER multi_regional_pivot_experiment.py completes.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Machine learning imports
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance

# DARTS imports (for loading pivoted data)
from darts import TimeSeries


def load_pivoted_data():
    """Load the pivoted regional weather data."""
    print("=" * 80)
    print("LOADING PIVOTED REGIONAL DATA")
    print("=" * 80)

    # Use same pivot logic as multi_regional_pivot_experiment.py
    from databricks import sql

    conn = sql.connect(
        server_hostname=os.environ['DATABRICKS_HOST'],
        http_path=os.environ['DATABRICKS_HTTP_PATH'],
        access_token=os.environ['DATABRICKS_TOKEN']
    )

    cursor = conn.cursor()
    query = """
        SELECT *
        FROM commodity.silver.unified_data
        WHERE commodity = 'Coffee'
        ORDER BY date, region
    """

    cursor.execute(query)
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]

    df = pd.DataFrame.from_records(rows, columns=columns)
    df['date'] = pd.to_datetime(df['date'])
    conn.close()

    # Pivot weather by region (same logic as experiment)
    weather_features = [
        'temp_max_c', 'temp_min_c', 'temp_mean_c',
        'precipitation_mm', 'rain_mm', 'snowfall_cm',
        'humidity_mean_pct', 'wind_speed_max_kmh'
    ]

    regions = sorted(df['region'].unique())
    print(f"Pivoting {len(regions)} regions...")

    pivoted_dfs = []
    for region in regions:
        df_region = df[df['region'] == region][['date'] + weather_features].copy()
        region_clean = region.lower().replace(' ', '_').replace(',', '')

        for col in weather_features:
            df_region[f'{col}_{region_clean}'] = df_region[col]
            df_region = df_region.drop(col, axis=1)

        pivoted_dfs.append(df_region)

    df_pivoted = pivoted_dfs[0]
    for region_df in pivoted_dfs[1:]:
        df_pivoted = df_pivoted.merge(region_df, on='date', how='outer')

    # Add target
    df_target = df.groupby('date')['close'].mean().reset_index()
    df_pivoted = df_pivoted.merge(df_target, on='date', how='left')

    # Forward-fill
    df_pivoted = df_pivoted.sort_values('date').fillna(method='ffill').fillna(0)

    print(f"Pivoted data shape: {df_pivoted.shape}")
    print(f"Features: {df_pivoted.shape[1] - 2} (excluding date and close)")

    return df_pivoted


def technique_1_tree_importance(X_train, y_train, feature_names, top_n=50):
    """Technique 1: XGBoost feature importance."""
    print("\n" + "=" * 80)
    print("TECHNIQUE 1: XGBoost Feature Importance")
    print("=" * 80)

    model = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )

    print("Training XGBoost...")
    model.fit(X_train, y_train)

    importances = model.feature_importances_
    feature_importance = list(zip(feature_names, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    print(f"\nTop {top_n} features by XGBoost importance:")
    for i, (feat, imp) in enumerate(feature_importance[:top_n], 1):
        print(f"  {i:2d}. {feat:50s} {imp:.6f}")

    return [feat for feat, _ in feature_importance[:top_n]]


def technique_2_random_forest(X_train, y_train, feature_names, top_n=50):
    """Technique 2: Random Forest feature importance."""
    print("\n" + "=" * 80)
    print("TECHNIQUE 2: Random Forest Feature Importance")
    print("=" * 80)

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    print("Training Random Forest...")
    model.fit(X_train, y_train)

    importances = model.feature_importances_
    feature_importance = list(zip(feature_names, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    print(f"\nTop {top_n} features by Random Forest importance:")
    for i, (feat, imp) in enumerate(feature_importance[:top_n], 1):
        print(f"  {i:2d}. {feat:50s} {imp:.6f}")

    return [feat for feat, _ in feature_importance[:top_n]]


def technique_3_correlation(X_train, y_train, feature_names, top_n=50):
    """Technique 3: Correlation analysis."""
    print("\n" + "=" * 80)
    print("TECHNIQUE 3: Correlation Analysis")
    print("=" * 80)

    # Calculate absolute correlations
    df_temp = pd.DataFrame(X_train, columns=feature_names)
    df_temp['target'] = y_train

    correlations = df_temp.corr()['target'].drop('target').abs()
    correlations.sort_values(ascending=False, inplace=True)

    print(f"\nTop {top_n} features by correlation:")
    for i, (feat, corr) in enumerate(correlations.head(top_n).items(), 1):
        print(f"  {i:2d}. {feat:50s} {corr:.6f}")

    return list(correlations.head(top_n).index)


def technique_4_permutation(X_train, y_train, X_val, y_val, feature_names, top_n=50):
    """Technique 4: Permutation importance."""
    print("\n" + "=" * 80)
    print("TECHNIQUE 4: Permutation Importance")
    print("=" * 80)

    # Train simple model for permutation
    model = XGBRegressor(n_estimators=50, max_depth=5, random_state=42)
    print("Training model for permutation test...")
    model.fit(X_train, y_train)

    print("Running permutation importance (this may take a few minutes)...")
    perm_importance = permutation_importance(
        model, X_val, y_val,
        n_repeats=5,
        scoring='neg_mean_absolute_percentage_error',
        random_state=42,
        n_jobs=-1
    )

    feature_importance = list(zip(feature_names, perm_importance.importances_mean))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    print(f"\nTop {top_n} features by permutation importance:")
    for i, (feat, imp) in enumerate(feature_importance[:top_n], 1):
        print(f"  {i:2d}. {feat:50s} {imp:.6f}")

    return [feat for feat, _ in feature_importance[:top_n]]


def technique_5_rfe(X_train, y_train, feature_names, top_n=50):
    """Technique 5: Recursive Feature Elimination."""
    print("\n" + "=" * 80)
    print("TECHNIQUE 5: Recursive Feature Elimination (RFE)")
    print("=" * 80)

    estimator = XGBRegressor(n_estimators=50, max_depth=5, random_state=42)
    selector = RFE(
        estimator,
        n_features_to_select=top_n,
        step=10,
        verbose=1
    )

    print("Running RFE (this may take a while)...")
    selector.fit(X_train, y_train)

    selected_features = [f for f, s in zip(feature_names, selector.support_) if s]

    print(f"\n{len(selected_features)} features selected by RFE:")
    for i, feat in enumerate(selected_features, 1):
        print(f"  {i:2d}. {feat}")

    return selected_features


def ensemble_voting(technique_results, min_votes=3):
    """Combine results from multiple techniques using ensemble voting."""
    print("\n" + "=" * 80)
    print("ENSEMBLE VOTING")
    print("=" * 80)

    vote_counts = Counter()
    for features in technique_results.values():
        for feat in features:
            vote_counts[feat] += 1

    print(f"\nVote distribution:")
    for votes in range(5, 0, -1):
        features_with_votes = [f for f, c in vote_counts.items() if c == votes]
        print(f"  {votes}/5 votes: {len(features_with_votes)} features")

    consensus_features = [f for f, count in vote_counts.items() if count >= min_votes]
    consensus_features.sort(key=lambda f: vote_counts[f], reverse=True)

    print(f"\n{len(consensus_features)} consensus features (≥{min_votes} votes):")
    for i, feat in enumerate(consensus_features[:50], 1):
        votes = vote_counts[feat]
        print(f"  {i:2d}. {feat:50s} [{votes}/5 votes]")

    return consensus_features


def analyze_by_region(consensus_features):
    """Analyze feature importance by region."""
    print("\n" + "=" * 80)
    print("REGIONAL IMPORTANCE ANALYSIS")
    print("=" * 80)

    # Extract region from feature name
    regional_counts = Counter()
    for feat in consensus_features:
        # Feature format: weather_var_region_name
        parts = feat.split('_')
        if len(parts) >= 3:
            # Region is everything after first weather var
            region = '_'.join(parts[2:]) if len(parts) > 2 else 'unknown'
            regional_counts[region] += 1

    print("\nTop regions by feature count:")
    for i, (region, count) in enumerate(regional_counts.most_common(10), 1):
        pct = count / len(consensus_features) * 100
        print(f"  {i:2d}. {region:40s} {count:3d} features ({pct:5.1f}%)")

    return regional_counts


def analyze_by_variable(consensus_features):
    """Analyze feature importance by weather variable."""
    print("\n" + "=" * 80)
    print("WEATHER VARIABLE IMPORTANCE ANALYSIS")
    print("=" * 80)

    variable_counts = Counter()
    for feat in consensus_features:
        # Extract weather variable (first part before region)
        for var in ['temp_max_c', 'temp_min_c', 'temp_mean_c',
                   'precipitation_mm', 'rain_mm', 'snowfall_cm',
                   'humidity_mean_pct', 'wind_speed_max_kmh']:
            if feat.startswith(var):
                variable_counts[var] += 1
                break

    print("\nWeather variables by feature count:")
    for i, (var, count) in enumerate(variable_counts.most_common(), 1):
        pct = count / len(consensus_features) * 100
        print(f"  {i}. {var:25s} {count:3d} features ({pct:5.1f}%)")

    return variable_counts


def run_feature_selection_analysis():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("FEATURE SELECTION ANALYSIS")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load data
    df = load_pivoted_data()

    # Prepare train/val split
    feature_cols = [c for c in df.columns if c not in ['date', 'close']]
    X = df[feature_cols].values
    y = df['close'].values

    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    print(f"\nTrain size: {len(X_train)}, Val size: {len(X_val)}")
    print(f"Features: {len(feature_cols)}")

    # Run all techniques
    technique_results = {}

    try:
        technique_results['xgboost'] = technique_1_tree_importance(
            X_train, y_train, feature_cols, top_n=50
        )
    except Exception as e:
        print(f"XGBoost failed: {e}")

    try:
        technique_results['random_forest'] = technique_2_random_forest(
            X_train, y_train, feature_cols, top_n=50
        )
    except Exception as e:
        print(f"Random Forest failed: {e}")

    try:
        technique_results['correlation'] = technique_3_correlation(
            X_train, y_train, feature_cols, top_n=50
        )
    except Exception as e:
        print(f"Correlation failed: {e}")

    try:
        technique_results['permutation'] = technique_4_permutation(
            X_train, y_train, X_val, y_val, feature_cols, top_n=50
        )
    except Exception as e:
        print(f"Permutation failed: {e}")

    # Skip RFE if too slow
    # try:
    #     technique_results['rfe'] = technique_5_rfe(
    #         X_train, y_train, feature_cols, top_n=50
    #     )
    # except Exception as e:
    #     print(f"RFE failed: {e}")

    # Ensemble voting
    consensus_features = ensemble_voting(technique_results, min_votes=2)

    # Dimensional analysis
    regional_counts = analyze_by_region(consensus_features)
    variable_counts = analyze_by_variable(consensus_features)

    # Save results
    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'num_original_features': len(feature_cols),
        'num_consensus_features': len(consensus_features),
        'consensus_features': consensus_features,
        'technique_results': {k: v[:20] for k, v in technique_results.items()},
        'top_regions': dict(regional_counts.most_common(10)),
        'top_variables': dict(variable_counts.most_common())
    }

    import json
    with open('feature_selection_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save consensus features as simple list
    with open('selected_features.txt', 'w') as f:
        for feat in consensus_features:
            f.write(feat + '\n')

    print("\n" + "=" * 80)
    print("RESULTS SAVED")
    print("=" * 80)
    print("  feature_selection_results.json - Full results")
    print("  selected_features.txt - List of selected features")
    print()
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    run_feature_selection_analysis()
