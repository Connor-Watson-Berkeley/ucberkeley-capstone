from setuptools import setup, find_packages

setup(
    name="ground_truth",
    version="0.4.1",
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
