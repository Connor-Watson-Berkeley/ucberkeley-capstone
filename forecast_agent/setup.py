from setuptools import setup, find_packages

setup(
    name="ground_truth",
    version="0.3.3",
    packages=find_packages() + ['utils'],
    package_dir={'utils': 'utils'},
    install_requires=[
        "pandas",
        "numpy",
        "xgboost",
        "statsmodels",
        "pmdarima"
    ]
)
