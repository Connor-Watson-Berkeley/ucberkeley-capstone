"""Upload distributions table to commodity.forecast.distributions in Databricks.

Usage:
    python upload_distributions_to_databricks.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from ground_truth.storage.databricks_writer import upload_distributions_to_databricks

if __name__ == "__main__":
    upload_distributions_to_databricks()
