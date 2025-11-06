import pandas as pd
import json

print('='*80)
print('FORECAST VALIDATION')
print('='*80)
print()

# Check point forecasts
print('1. Point Forecasts:')
print('-'*80)
df = pd.read_parquet('production_forecasts/point_forecasts.parquet')
print('Rows by commodity:')
print(df.groupby(['commodity', 'model_version']).size())
print()

# Show Sugar forecasts
print('Sample Sugar forecasts:')
sugar = df[df['commodity'] == 'Sugar'].sort_values('forecast_date')
print(sugar[['forecast_date', 'day_ahead', 'forecast_mean', 'lower_95', 'upper_95']].head(7).to_string(index=False))
print()

# Show Coffee for comparison
print('Sample Coffee forecasts (latest model):')
coffee = df[(df['commodity'] == 'Coffee') & (df['model_version'] == 'sarimax_weather_v1_coffee')].sort_values('forecast_date')
print(coffee[['forecast_date', 'day_ahead', 'forecast_mean', 'lower_95', 'upper_95']].head(7).to_string(index=False))
print()

# Check distributions
print('2. Distributions:')
print('-'*80)
df_dist = pd.read_parquet('production_forecasts/distributions.parquet')
print(f'Total rows: {len(df_dist):,}')
print('Paths per commodity:')
print(df_dist.groupby('commodity')['path_id'].nunique())
print()

# Check JSON exports
print('3. Trading Agent Exports:')
print('-'*80)
import os
for commodity_name in ['coffee', 'sugar']:
    json_file = f'trading_agent_forecast_{commodity_name}.json'
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        print(f'{commodity_name.title()}:')
        print(f'  Model: {data["model_version"]}')
        print(f'  Forecasts: {len(data["forecasts"])} days')
        if len(data["forecasts"]) > 0:
            print(f'  Day 1: ${data["forecasts"][0]["forecast_value"]:.2f}')
            print(f'  Day 7: ${data["forecasts"][6]["forecast_value"]:.2f}')
            print(f'  Day 14: ${data["forecasts"][13]["forecast_value"]:.2f}')
        print()
print('='*80)
