"""
Statistical Validation for Trading Strategies

Provides statistical tests to determine if strategy improvements are significant
or could be due to random chance.

Key tests:
- Paired t-test: Does strategy beat baseline across years?
- Sign test: Non-parametric robustness check
- Bootstrap CI: Confidence intervals for differences
- Effect sizes: Cohen's d for practical significance
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple, List, Optional, Any


class StatisticalAnalyzer:
    """
    Runs statistical validation on backtest results

    Answers the key question: "Is this improvement statistically significant
    or could it be random chance?"
    """

    def __init__(self, spark=None):
        """
        Initialize statistical analyzer

        Args:
            spark: Spark session (required for loading Delta tables)
        """
        self.spark = spark

    def load_year_by_year_results(
        self,
        commodity: str,
        model_version: str
    ) -> pd.DataFrame:
        """
        Load year-by-year results from Delta table

        Args:
            commodity: Commodity name (e.g., 'coffee')
            model_version: Model version (e.g., 'naive')

        Returns:
            DataFrame with columns [year, strategy, net_earnings, ...]
        """
        if self.spark is None:
            raise ValueError("Spark session required to load results")

        table_name = f"commodity.trading_agent.results_{commodity}_by_year_{model_version}"

        try:
            df = self.spark.table(table_name).toPandas()
            print(f"✓ Loaded {len(df)} year-strategy combinations from {table_name}")
            return df
        except Exception as e:
            raise ValueError(f"Could not load results table {table_name}: {e}")

    def run_full_analysis(
        self,
        commodity: str,
        model_version: str,
        primary_baseline: str = "Immediate Sale",
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete statistical analysis for a commodity-model combination

        Args:
            commodity: Commodity name
            model_version: Model version
            primary_baseline: Baseline strategy to compare against
            verbose: Print detailed results

        Returns:
            Dict with all statistical test results
        """
        if verbose:
            print("\n" + "=" * 80)
            print(f"STATISTICAL ANALYSIS: {commodity.upper()} - {model_version}")
            print("=" * 80)

        # Load data
        year_df = self.load_year_by_year_results(commodity, model_version)

        # Get all strategies
        strategies = year_df['strategy'].unique().tolist()

        # Identify prediction strategies
        prediction_strategies = [
            s for s in strategies if any([
                'Predictive' in s,
                s in ['Consensus', 'Expected Value', 'Risk-Adjusted', 'Rolling Horizon MPC']
            ])
        ]

        baseline_strategies = [s for s in strategies if s not in prediction_strategies]

        if verbose:
            print(f"\n✓ Found {len(strategies)} strategies:")
            print(f"  • {len(baseline_strategies)} baselines: {baseline_strategies}")
            print(f"  • {len(prediction_strategies)} prediction-based: {prediction_strategies}")
            print(f"\nPrimary baseline: {primary_baseline}")

        results = {
            'commodity': commodity,
            'model_version': model_version,
            'primary_baseline': primary_baseline,
            'n_strategies': len(strategies),
            'n_years': len(year_df['year'].unique()),
            'years': sorted(year_df['year'].unique().tolist()),
            'strategy_vs_baseline_tests': [],
            'matched_pair_tests': [],
            'best_prediction_analysis': None
        }

        # Test each prediction strategy vs primary baseline
        if verbose:
            print(f"\n{'=' * 80}")
            print(f"PREDICTION STRATEGIES vs {primary_baseline.upper()}")
            print("=" * 80)

        for strategy in prediction_strategies:
            test_result = test_strategy_vs_baseline(
                strategy_name=strategy,
                baseline_name=primary_baseline,
                year_df=year_df,
                verbose=verbose
            )
            results['strategy_vs_baseline_tests'].append(test_result)

        # Test matched pairs (forecast integration benefit)
        matched_pairs = [
            ('Price Threshold Predictive', 'Price Threshold'),
            ('Moving Average Predictive', 'Moving Average')
        ]

        if verbose:
            print(f"\n{'=' * 80}")
            print("MATCHED PAIR TESTS (Forecast Integration Benefit)")
            print("=" * 80)

        for pred_strategy, base_strategy in matched_pairs:
            if pred_strategy in strategies and base_strategy in strategies:
                test_result = test_strategy_vs_baseline(
                    strategy_name=pred_strategy,
                    baseline_name=base_strategy,
                    year_df=year_df,
                    verbose=verbose
                )
                results['matched_pair_tests'].append(test_result)

        # Identify best prediction strategy and analyze it
        if results['strategy_vs_baseline_tests']:
            best_pred = max(
                results['strategy_vs_baseline_tests'],
                key=lambda x: x.get('mean_difference', float('-inf'))
            )
            results['best_prediction_analysis'] = best_pred

            if verbose:
                print(f"\n{'=' * 80}")
                print("BEST PREDICTION STRATEGY")
                print("=" * 80)
                self._print_detailed_analysis(best_pred)

        return results

    def _print_detailed_analysis(self, test_result: Dict) -> None:
        """Print detailed analysis of a test result"""
        print(f"\n🏆 {test_result['strategy']} vs {test_result['baseline']}")
        print(f"\nSample Size: {test_result['n_years']} years ({test_result['years'][0]}-{test_result['years'][-1]})")

        print(f"\nDescriptive Statistics:")
        print(f"  Mean earnings ({test_result['strategy']}): ${test_result['mean_strategy']:,.0f}")
        print(f"  Mean earnings ({test_result['baseline']}): ${test_result['mean_baseline']:,.0f}")
        print(f"  Mean difference: ${test_result['mean_difference']:,.0f}")
        print(f"  Std of differences: ${test_result['std_difference']:,.0f}")

        print(f"\nPaired t-test:")
        print(f"  t-statistic: {test_result['t_statistic']:.4f}")
        print(f"  p-value: {test_result['p_value']:.4f}", end="")
        if test_result['significant_05']:
            print(" ✓ SIGNIFICANT at α=0.05")
        elif test_result['p_value'] < 0.10:
            print(" ⚠️  Marginally significant (p<0.10)")
        else:
            print(" ✗ Not significant")

        print(f"\nEffect Size:")
        print(f"  Cohen's d: {test_result['cohens_d']:.4f} ({test_result['effect_interpretation']})")

        print(f"\n95% Confidence Interval:")
        print(f"  [{test_result['ci_95_lower']:,.0f}, {test_result['ci_95_upper']:,.0f}]")
        if test_result['ci_includes_zero']:
            print("  ⚠️  Includes zero (cannot rule out no effect)")
        else:
            print("  ✓ Does not include zero")

        print(f"\nSign Test (Non-Parametric):")
        print(f"  Years positive: {test_result['n_years_positive']}/{test_result['n_years']}")
        print(f"  Years negative: {test_result['n_years_negative']}/{test_result['n_years']}")
        print(f"  p-value: {test_result['sign_test_p_value']:.4f}", end="")
        if test_result['sign_test_significant']:
            print(" ✓ SIGNIFICANT")
        else:
            print(" ✗ Not significant")

    def save_results(
        self,
        results: Dict[str, Any],
        save_to_delta: bool = True
    ) -> Optional[str]:
        """
        Save statistical test results to Delta table

        Args:
            results: Output from run_full_analysis()
            save_to_delta: If True, save to Delta table

        Returns:
            Table name if saved, None otherwise
        """
        if not save_to_delta or self.spark is None:
            return None

        commodity = results['commodity']
        model_version = results['model_version']

        # Flatten all test results to DataFrame
        all_tests = []

        # Strategy vs baseline tests
        for test in results['strategy_vs_baseline_tests']:
            test_copy = test.copy()
            test_copy['test_type'] = 'strategy_vs_baseline'
            test_copy['years_list'] = str(test_copy['years'])  # Convert list to string for Delta
            all_tests.append(test_copy)

        # Matched pair tests
        for test in results['matched_pair_tests']:
            test_copy = test.copy()
            test_copy['test_type'] = 'matched_pair'
            test_copy['years_list'] = str(test_copy['years'])
            all_tests.append(test_copy)

        if not all_tests:
            print("⚠️  No test results to save")
            return None

        stats_df = pd.DataFrame(all_tests)

        # Add metadata
        stats_df['commodity'] = commodity
        stats_df['model_version'] = model_version
        stats_df['analysis_timestamp'] = pd.Timestamp.now()

        # Remove list column (converted to string above)
        if 'years' in stats_df.columns:
            stats_df = stats_df.drop(columns=['years'])

        # Save to Delta
        table_name = f"commodity.trading_agent.statistical_tests_{commodity}_{model_version}"

        try:
            self.spark.createDataFrame(stats_df).write \
                .mode("overwrite") \
                .option("overwriteSchema", "true") \
                .saveAsTable(table_name)

            print(f"\n✓ Saved statistical tests to: {table_name}")
            return table_name

        except Exception as e:
            print(f"\n❌ Error saving to Delta: {e}")
            return None


def test_strategy_vs_baseline(
    strategy_name: str,
    baseline_name: str,
    year_df: pd.DataFrame,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Test if strategy significantly beats baseline using paired t-test

    This is a paired test because we compare the SAME years for both strategies,
    controlling for year-to-year market variation.

    Args:
        strategy_name: Strategy to test (e.g., "Rolling Horizon MPC")
        baseline_name: Baseline to compare against (e.g., "Immediate Sale")
        year_df: DataFrame with columns [year, strategy, net_earnings]
        verbose: Print results

    Returns:
        Dict with comprehensive statistical test results
    """
    # Get year-by-year earnings for each strategy
    strategy_years = year_df[year_df['strategy'] == strategy_name].set_index('year')['net_earnings']
    baseline_years = year_df[year_df['strategy'] == baseline_name].set_index('year')['net_earnings']

    # Find overlapping years
    common_years = strategy_years.index.intersection(baseline_years.index)

    if len(common_years) < 3:
        result = {
            'error': f'Insufficient overlapping years: {len(common_years)}',
            'strategy': strategy_name,
            'baseline': baseline_name,
            'n_years': len(common_years)
        }
        if verbose:
            print(f"\n⚠️  {strategy_name} vs {baseline_name}: Only {len(common_years)} overlapping years (need ≥3)")
        return result

    # Align data by year (preserves pairing)
    strategy_values = strategy_years.loc[common_years].values
    baseline_values = baseline_years.loc[common_years].values
    differences = strategy_values - baseline_values

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(strategy_values, baseline_values)

    # Effect size (Cohen's d for paired data)
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0

    # 95% Confidence interval for mean difference
    ci = stats.t.interval(
        confidence=0.95,
        df=len(differences) - 1,
        loc=mean_diff,
        scale=stats.sem(differences)
    )

    # Sign test (non-parametric alternative)
    n_positive = np.sum(differences > 0)
    n_total = len(differences)
    # Binomial test: H0 = 50% probability of being positive
    # Use binomtest (binom_test was deprecated)
    sign_p_value = stats.binomtest(n_positive, n_total, 0.5, alternative='greater').pvalue

    # Bootstrap confidence interval for robustness
    boot_ci = bootstrap_confidence_interval(strategy_values, baseline_values, n_bootstrap=10000)

    result = {
        'strategy': strategy_name,
        'baseline': baseline_name,
        'n_years': len(common_years),
        'years': sorted(common_years.tolist()),

        # Descriptive statistics
        'mean_strategy': float(np.mean(strategy_values)),
        'mean_baseline': float(np.mean(baseline_values)),
        'mean_difference': float(mean_diff),
        'std_difference': float(std_diff),

        # Paired t-test
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant_05': bool(p_value < 0.05),
        'significant_01': bool(p_value < 0.01),

        # Effect size
        'cohens_d': float(cohens_d),
        'effect_interpretation': _interpret_cohens_d(cohens_d),

        # Parametric confidence interval
        'ci_95_lower': float(ci[0]),
        'ci_95_upper': float(ci[1]),
        'ci_includes_zero': bool(ci[0] <= 0 <= ci[1]),

        # Bootstrap confidence interval
        'bootstrap_ci_95_lower': float(boot_ci[0]),
        'bootstrap_ci_95_upper': float(boot_ci[1]),

        # Sign test
        'n_years_positive': int(n_positive),
        'n_years_negative': int(n_total - n_positive),
        'sign_test_p_value': float(sign_p_value),
        'sign_test_significant': bool(sign_p_value < 0.05)
    }

    if verbose:
        print(f"\n{strategy_name} vs {baseline_name}:")
        print(f"  n={len(common_years)} years, Δ=${mean_diff:,.0f}, p={p_value:.4f}", end="")
        if p_value < 0.05:
            print(" ✓")
        elif p_value < 0.10:
            print(" ⚠️")
        else:
            print(" ✗")

    return result


def bootstrap_confidence_interval(
    strategy_earnings: np.ndarray,
    baseline_earnings: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Bootstrap confidence interval for difference in means

    Non-parametric alternative to t-test confidence interval.
    Doesn't assume normal distribution of differences.

    Args:
        strategy_earnings: Array of strategy earnings by year
        baseline_earnings: Array of baseline earnings by year (same length)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (0.95 = 95%)

    Returns:
        (lower_bound, upper_bound)
    """
    n = len(strategy_earnings)
    differences = []

    for _ in range(n_bootstrap):
        # Resample with replacement (preserves pairing)
        indices = np.random.choice(n, size=n, replace=True)
        boot_diff = np.mean(strategy_earnings[indices] - baseline_earnings[indices])
        differences.append(boot_diff)

    alpha = 1 - confidence
    lower = np.percentile(differences, alpha/2 * 100)
    upper = np.percentile(differences, (1 - alpha/2) * 100)

    return (lower, upper)


def run_full_statistical_analysis(
    spark,
    commodity: str,
    model_version: str,
    primary_baseline: str = "Immediate Sale",
    verbose: bool = True,
    save_to_delta: bool = True
) -> Dict[str, Any]:
    """
    Convenience function: Run complete statistical analysis

    Args:
        spark: Spark session
        commodity: Commodity name
        model_version: Model version
        primary_baseline: Baseline strategy name
        verbose: Print detailed results
        save_to_delta: Save results to Delta table

    Returns:
        Dict with all statistical test results
    """
    analyzer = StatisticalAnalyzer(spark=spark)
    results = analyzer.run_full_analysis(
        commodity=commodity,
        model_version=model_version,
        primary_baseline=primary_baseline,
        verbose=verbose
    )

    if save_to_delta:
        analyzer.save_results(results, save_to_delta=True)

    return results


def _interpret_cohens_d(d: float) -> str:
    """
    Interpret Cohen's d effect size

    Standard interpretation:
    - |d| < 0.2: negligible
    - 0.2 ≤ |d| < 0.5: small
    - 0.5 ≤ |d| < 0.8: medium
    - |d| ≥ 0.8: large
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"
