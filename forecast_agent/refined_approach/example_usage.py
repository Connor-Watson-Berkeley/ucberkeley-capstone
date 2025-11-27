"""Example usage of refined approach for forecast agent.

Demonstrates the streamlined OOP design inspired by DS261.
"""

from pyspark.sql import SparkSession
# Support both relative imports (package) and absolute imports (script)
try:
    from .data_loader import TimeSeriesDataLoader
    from .evaluator import ForecastEvaluator
    from .cross_validator import TimeSeriesCrossValidator
    from .model_pipeline import create_model_from_registry, NaivePipeline, RandomWalkPipeline
except ImportError:
    # Running as script
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from refined_approach.data_loader import TimeSeriesDataLoader
    from refined_approach.evaluator import ForecastEvaluator
    from refined_approach.cross_validator import TimeSeriesCrossValidator
    from refined_approach.model_pipeline import create_model_from_registry, NaivePipeline, RandomWalkPipeline


def example_basic_usage():
    """Basic example: Load data, create folds, run cross-validation."""
    
    # Initialize Spark (if running in Databricks)
    try:
        spark = SparkSession.builder.getOrCreate()
    except:
        spark = None  # Running locally without Spark
    
    # 1. Create data loader
    loader = TimeSeriesDataLoader(spark=spark)
    
    # 2. Load data to pandas
    df = loader.load_to_pandas(
        commodity='Coffee',
        cutoff_date='2024-01-01',
        features=['close', 'temp_mean_c', 'humidity_mean_pct']
    )
    
    # 3. Create temporal folds
    folds = loader.create_walk_forward_folds(
        df=df,
        min_train_size=365,
        step_size=14,
        horizon=14,
        max_folds=10  # Limit for example
    )
    
    print(f"Created {len(folds)} folds")
    
    # 4. Create evaluator
    evaluator = ForecastEvaluator(target_col='close', prediction_col='forecast')
    
    # 5. Create cross-validator
    cv = TimeSeriesCrossValidator(
        data_loader=loader,
        evaluator=evaluator,
        folds=folds
    )
    
    # 6. Run cross-validation with naive model
    naive_model = NaivePipeline()
    
    metrics_df = cv.fit(
        model_fn=naive_model,
        model_params={},
        target_col='close',
        horizon=14
    )
    
    print("\nCross-validation metrics:")
    print(metrics_df)
    
    # 7. Evaluate on test fold
    test_metrics = cv.evaluate_test(
        model_fn=naive_model,
        model_params={},
        target_col='close',
        horizon=14
    )
    
    print("\nTest metrics:")
    print(test_metrics)
    
    return cv, metrics_df, test_metrics


def example_model_comparison():
    """Compare multiple models using cross-validation."""
    
    # Setup (same as above)
    try:
        spark = SparkSession.builder.getOrCreate()
    except:
        spark = None
    
    loader = TimeSeriesDataLoader(spark=spark)
    df = loader.load_to_pandas(commodity='Coffee', cutoff_date='2024-01-01')
    
    folds = loader.create_walk_forward_folds(df, min_train_size=365, step_size=14, horizon=14, max_folds=5)
    
    evaluator = ForecastEvaluator()
    
    # Test multiple models
    models = {
        'naive': NaivePipeline(),
        'random_walk': RandomWalkPipeline(lookback_days=30)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Evaluating {name}")
        print(f"{'='*60}")
        
        cv = TimeSeriesCrossValidator(loader, evaluator, folds)
        metrics_df = cv.fit(model_fn=model, model_params={}, target_col='close', horizon=14)
        
        results[name] = cv
        
        print(f"\n{name} CV Results:")
        print(metrics_df.tail())  # Show summary rows
    
    # Compare models
    print(f"\n{'='*60}")
    print("Model Comparison")
    print(f"{'='*60}")
    
    comparison = TimeSeriesCrossValidator.compare_models(
        results['naive'],
        results['random_walk'],
        name1="Naive",
        name2="Random Walk"
    )
    
    print(f"MAE Improvement: {comparison.get('mae_improvement_pct', 0):.2f}%")
    print(f"Dir Day0 - Naive: {comparison.get('model1_dir_day0_mean', 0):.2f}%")
    print(f"Dir Day0 - Random Walk: {comparison.get('model2_dir_day0_mean', 0):.2f}%")
    
    return results, comparison


def example_with_registry_model():
    """Use existing model registry models with refined approach."""
    
    try:
        spark = SparkSession.builder.getOrCreate()
    except:
        spark = None
    
    loader = TimeSeriesDataLoader(spark=spark)
    df = loader.load_to_pandas(commodity='Coffee', cutoff_date='2024-01-01')
    
    folds = loader.create_walk_forward_folds(df, min_train_size=365, step_size=14, horizon=14, max_folds=5)
    
    evaluator = ForecastEvaluator()
    
    # Create model from registry
    model = create_model_from_registry('naive')
    
    cv = TimeSeriesCrossValidator(loader, evaluator, folds)
    metrics_df = cv.fit(model_fn=model, model_params={}, target_col='close', horizon=14)
    
    print("Registry model results:")
    print(metrics_df)
    
    return cv, metrics_df


if __name__ == "__main__":
    print("Running refined approach examples...")
    print("\nExample 1: Basic Usage")
    print("-" * 60)
    
    try:
        cv, metrics, test = example_basic_usage()
        print("\n✅ Basic example completed")
    except Exception as e:
        print(f"\n❌ Basic example failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n\nExample 2: Model Comparison")
    print("-" * 60)
    
    try:
        results, comparison = example_model_comparison()
        print("\n✅ Comparison example completed")
    except Exception as e:
        print(f"\n❌ Comparison example failed: {e}")
        import traceback
        traceback.print_exc()

