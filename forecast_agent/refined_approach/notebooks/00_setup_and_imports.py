"""
Setup cell for Databricks notebooks - handles imports with fail-open behavior.

Import this in notebooks to ensure package loading failures don't crash the job.
"""

# COMMAND ----------

# MAGIC %md
# MAGIC # Setup: Import Modules with Fail-Open Behavior
# MAGIC 
# MAGIC This cell handles imports safely - if a package fails to load, we log it and continue.

# COMMAND ----------

import sys
from pathlib import Path

# Add refined_approach to path
workspace_path = Path('/Workspace/Repos')
repo_path = None
for repo in workspace_path.iterdir():
    if repo.is_dir():
        refined_path = repo / 'forecast_agent' / 'refined_approach'
        if refined_path.exists():
            repo_path = refined_path
            break

if repo_path:
    sys.path.insert(0, str(repo_path))
    print(f"✅ Added {repo_path} to path")
else:
    # Fallback: assume current directory structure
    sys.path.insert(0, str(Path.cwd() / 'forecast_agent' / 'refined_approach'))
    print("⚠️  Using fallback path")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Core Modules (Required)

# COMMAND ----------

# Core modules - these should always be available
try:
    from data_loader import TimeSeriesDataLoader
    print("✅ TimeSeriesDataLoader imported")
except ImportError as e:
    print(f"❌ Failed to import TimeSeriesDataLoader: {e}")
    raise  # Core module - raise if missing

try:
    from evaluator import ForecastEvaluator
    print("✅ ForecastEvaluator imported")
except ImportError as e:
    print(f"❌ Failed to import ForecastEvaluator: {e}")
    raise  # Core module - raise if missing

try:
    from model_pipeline import ModelPipeline, create_model_from_registry
    print("✅ ModelPipeline imported")
except ImportError as e:
    print(f"❌ Failed to import ModelPipeline: {e}")
    raise  # Core module - raise if missing

try:
    from model_persistence import save_model_spark, load_model_spark, model_exists_spark
    print("✅ Model persistence imported")
except ImportError as e:
    print(f"❌ Failed to import model_persistence: {e}")
    raise  # Core module - raise if missing

try:
    from distributions_writer import DistributionsWriter, get_existing_forecast_dates
    print("✅ DistributionsWriter imported")
except ImportError as e:
    print(f"❌ Failed to import distributions_writer: {e}")
    raise  # Core module - raise if missing

try:
    from daily_production import should_retrain_today, get_most_recent_trained_model
    print("✅ Daily production utilities imported")
except ImportError as e:
    print(f"⚠️  Daily production utilities not available: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Registry (Fail-Open for Package Dependencies)

# COMMAND ----------

# Try to import model registry - may fail if ground_truth package not installed
MODEL_REGISTRY_AVAILABLE = False
BASELINE_MODELS = {}

try:
    # Try importing from parent directory (if ground_truth is available)
    parent_path = str(Path(repo_path).parent.parent) if repo_path else str(Path.cwd() / 'forecast_agent')
    sys.path.insert(0, parent_path)
    
    from ground_truth.config.model_registry import BASELINE_MODELS
    MODEL_REGISTRY_AVAILABLE = True
    print(f"✅ Model registry imported ({len(BASELINE_MODELS)} models available)")
    
except ImportError as e:
    print(f"⚠️  Could not import model registry: {e}")
    print("   Will use simplified model implementations only")
    
    # Define minimal model registry for basic models
    BASELINE_MODELS = {
        'naive': {
            'name': 'Naive',
            'function': None,  # Will use ModelPipeline implementation
            'params': {'target': 'close', 'horizon': 14}
        },
        'random_walk': {
            'name': 'RandomWalk',
            'function': None,
            'params': {'target': 'close', 'horizon': 14, 'lookback_days': 30}
        }
    }
    print(f"   Using minimal model registry ({len(BASELINE_MODELS)} models)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Import Status

# COMMAND ----------

# Check which models can be loaded (fail-open per model)
AVAILABLE_MODELS = []
FAILED_MODELS = []

if MODEL_REGISTRY_AVAILABLE:
    print("\n📦 Checking model availability...")
    
    for model_key in BASELINE_MODELS.keys():
        try:
            model = create_model_from_registry(model_key)
            AVAILABLE_MODELS.append(model_key)
            print(f"   ✅ {model_key}: Available")
        except Exception as e:
            FAILED_MODELS.append((model_key, str(e)[:100]))
            print(f"   ⚠️  {model_key}: Failed ({str(e)[:50]}...)")
    
    print(f"\n📊 Summary: {len(AVAILABLE_MODELS)} available, {len(FAILED_MODELS)} failed")
    
    if AVAILABLE_MODELS:
        print(f"\n✅ Available models: {', '.join(AVAILABLE_MODELS)}")
    
    if FAILED_MODELS:
        print(f"\n⚠️  Failed models (will be skipped):")
        for model_key, error in FAILED_MODELS:
            print(f"   - {model_key}: {error}")
else:
    print("\n⚠️  Model registry not available - using basic models only")
    AVAILABLE_MODELS = ['naive', 'random_walk']  # Always available (no dependencies)

# COMMAND ----------

