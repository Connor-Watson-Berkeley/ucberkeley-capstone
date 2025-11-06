# Databricks notebook source
# MAGIC %md
# MAGIC # Ground Truth - Forecast Performance Dashboard
# MAGIC
# MAGIC Interactive dashboard for monitoring commodity price forecast performance.
# MAGIC
# MAGIC **Features:**
# MAGIC - Compare forecast models side-by-side
# MAGIC - Analyze forecast accuracy over time
# MAGIC - Visualize forecast vs actual prices
# MAGIC - Examine forecast error distributions
# MAGIC - Track model performance metrics

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup & Imports

# COMMAND ----------

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Interactive Controls

# COMMAND ----------

# Create widgets for user interaction
dbutils.widgets.dropdown(
    "commodity",
    "Coffee",
    ["All", "Coffee", "Sugar"],
    "1. Select Commodity"
)

dbutils.widgets.dropdown(
    "model",
    "All",
    ["All"],  # Will be populated dynamically
    "2. Select Model"
)

dbutils.widgets.dropdown(
    "horizon",
    "All",
    ["All", "1-day", "3-day", "7-day", "14-day"],
    "3. Forecast Horizon"
)

dbutils.widgets.dropdown(
    "metric",
    "MAE",
    ["MAE", "RMSE", "MAPE", "Directional Accuracy"],
    "4. Primary Metric"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

# Load forecast data
df_forecasts = spark.read.parquet('/FileStore/production_forecasts/point_forecasts.parquet').toPandas()
df_actuals = spark.read.parquet('/FileStore/production_forecasts/forecast_actuals.parquet').toPandas()

# Convert dates
df_forecasts['forecast_date'] = pd.to_datetime(df_forecasts['forecast_date'])
df_forecasts['data_cutoff_date'] = pd.to_datetime(df_forecasts['data_cutoff_date'])
df_actuals['forecast_date'] = pd.to_datetime(df_actuals['forecast_date'])

print(f"✅ Loaded {len(df_forecasts):,} forecasts")
print(f"✅ Loaded {len(df_actuals):,} actual observations")

# Update model dropdown with available models
available_models = ["All"] + sorted(df_forecasts['model_version'].unique().tolist())
dbutils.widgets.remove("model")
dbutils.widgets.dropdown("model", "All", available_models, "2. Select Model")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Preparation

# COMMAND ----------

# Get widget values
selected_commodity = dbutils.widgets.get("commodity")
selected_model = dbutils.widgets.get("model")
selected_horizon = dbutils.widgets.get("horizon")
selected_metric = dbutils.widgets.get("metric")

# Filter forecasts
df_filt = df_forecasts.copy()

if selected_commodity != "All":
    df_filt = df_filt[df_filt['commodity'] == selected_commodity]

if selected_model != "All":
    df_filt = df_filt[df_filt['model_version'] == selected_model]

# Filter by horizon
if selected_horizon != "All":
    horizon_map = {"1-day": 1, "3-day": 3, "7-day": 7, "14-day": 14}
    df_filt = df_filt[df_filt['day_ahead'] == horizon_map[selected_horizon]]

# Join with actuals to calculate errors
df_eval = df_filt.merge(
    df_actuals[['commodity', 'forecast_date', 'actual_close']],
    on=['commodity', 'forecast_date'],
    how='inner'
)

# Calculate error metrics
df_eval['error'] = df_eval['forecast_mean'] - df_eval['actual_close']
df_eval['abs_error'] = np.abs(df_eval['error'])
df_eval['squared_error'] = df_eval['error'] ** 2
df_eval['pct_error'] = (df_eval['error'] / df_eval['actual_close']) * 100
df_eval['abs_pct_error'] = np.abs(df_eval['pct_error'])

# Directional accuracy (did we predict direction correctly?)
df_eval_dir = df_eval.merge(
    df_actuals[['commodity', 'forecast_date', 'actual_close']].rename(
        columns={'forecast_date': 'prev_date', 'actual_close': 'prev_actual'}
    ),
    left_on=['commodity', df_eval['forecast_date'] - pd.Timedelta(days=1)],
    right_on=['commodity', 'prev_date'],
    how='left'
)
df_eval['forecast_direction'] = np.sign(df_eval['forecast_mean'] - df_eval_dir['prev_actual'])
df_eval['actual_direction'] = np.sign(df_eval['actual_close'] - df_eval_dir['prev_actual'])
df_eval['direction_correct'] = (df_eval['forecast_direction'] == df_eval['actual_direction']).astype(int)

print(f"📊 Evaluating {len(df_eval):,} forecasts with actuals")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📈 Key Performance Metrics

# COMMAND ----------

# Calculate overall metrics
mae = df_eval['abs_error'].mean()
rmse = np.sqrt(df_eval['squared_error'].mean())
mape = df_eval['abs_pct_error'].mean()
directional_acc = df_eval['direction_correct'].mean() * 100

# Display as metric cards
displayHTML(f"""
<style>
    .metric-container {{
        display: flex;
        justify-content: space-around;
        margin: 20px 0;
    }}
    .metric-card {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        color: white;
        min-width: 200px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    .metric-value {{
        font-size: 36px;
        font-weight: bold;
        margin: 10px 0;
    }}
    .metric-label {{
        font-size: 14px;
        opacity: 0.9;
    }}
</style>

<div class="metric-container">
    <div class="metric-card">
        <div class="metric-label">Mean Absolute Error</div>
        <div class="metric-value">${mae:.2f}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Root Mean Squared Error</div>
        <div class="metric-value">${rmse:.2f}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Mean Abs Pct Error</div>
        <div class="metric-value">{mape:.1f}%</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Directional Accuracy</div>
        <div class="metric-value">{directional_acc:.1f}%</div>
    </div>
</div>
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Forecast vs Actual - Time Series

# COMMAND ----------

# Create forecast vs actual plot
fig = go.Figure()

# Group by commodity and model for plotting
for commodity in df_eval['commodity'].unique():
    for model in df_eval['model_version'].unique():
        df_plot = df_eval[
            (df_eval['commodity'] == commodity) &
            (df_eval['model_version'] == model)
        ].sort_values('forecast_date')

        # Plot actuals
        fig.add_trace(go.Scatter(
            x=df_plot['forecast_date'],
            y=df_plot['actual_close'],
            mode='lines',
            name=f'{commodity} - Actual',
            line=dict(color='black', width=2),
            showlegend=True
        ))

        # Plot forecasts
        fig.add_trace(go.Scatter(
            x=df_plot['forecast_date'],
            y=df_plot['forecast_mean'],
            mode='markers+lines',
            name=f'{commodity} - {model[:20]}',
            line=dict(width=2, dash='dot'),
            marker=dict(size=6),
            showlegend=True
        ))

fig.update_layout(
    title="Forecast vs Actual Prices",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    height=500,
    hovermode='x unified',
    template='plotly_white'
)

fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📉 Forecast Error Over Time

# COMMAND ----------

# Plot error over time
fig = go.Figure()

for commodity in df_eval['commodity'].unique():
    for model in df_eval['model_version'].unique():
        df_plot = df_eval[
            (df_eval['commodity'] == commodity) &
            (df_eval['model_version'] == model)
        ].sort_values('forecast_date')

        fig.add_trace(go.Scatter(
            x=df_plot['forecast_date'],
            y=df_plot['error'],
            mode='markers',
            name=f'{commodity} - {model[:20]}',
            marker=dict(size=6, opacity=0.6)
        ))

# Add zero line
fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Zero Error")

fig.update_layout(
    title="Forecast Error Over Time (Positive = Over-predicted)",
    xaxis_title="Date",
    yaxis_title="Error (USD)",
    height=400,
    hovermode='x unified',
    template='plotly_white'
)

fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Error Distribution

# COMMAND ----------

# Create error distribution histogram
fig = go.Figure()

for commodity in df_eval['commodity'].unique():
    for model in df_eval['model_version'].unique():
        df_plot = df_eval[
            (df_eval['commodity'] == commodity) &
            (df_eval['model_version'] == model)
        ]

        fig.add_trace(go.Histogram(
            x=df_plot['error'],
            name=f'{commodity} - {model[:20]}',
            opacity=0.7,
            nbinsx=30
        ))

fig.update_layout(
    title="Forecast Error Distribution",
    xaxis_title="Error (USD)",
    yaxis_title="Frequency",
    height=400,
    barmode='overlay',
    template='plotly_white'
)

fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📈 Performance by Forecast Horizon

# COMMAND ----------

# Calculate metrics by horizon
horizon_metrics = df_eval.groupby(['commodity', 'model_version', 'day_ahead']).agg({
    'abs_error': 'mean',
    'squared_error': lambda x: np.sqrt(x.mean()),
    'abs_pct_error': 'mean',
    'direction_correct': lambda x: x.mean() * 100
}).reset_index()

horizon_metrics.columns = ['commodity', 'model_version', 'day_ahead', 'MAE', 'RMSE', 'MAPE', 'Directional_Accuracy']

# Plot MAE by horizon
fig = px.line(
    horizon_metrics,
    x='day_ahead',
    y='MAE',
    color='model_version',
    facet_col='commodity',
    markers=True,
    title='Forecast Error by Horizon (MAE)',
    labels={'day_ahead': 'Days Ahead', 'MAE': 'Mean Absolute Error (USD)'}
)

fig.update_layout(height=400, template='plotly_white')
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎯 Model Comparison Table

# COMMAND ----------

# Calculate metrics by model and commodity
model_comparison = df_eval.groupby(['commodity', 'model_version']).agg({
    'abs_error': 'mean',
    'squared_error': lambda x: np.sqrt(x.mean()),
    'abs_pct_error': 'mean',
    'direction_correct': lambda x: x.mean() * 100,
    'forecast_date': 'count'
}).reset_index()

model_comparison.columns = [
    'Commodity', 'Model', 'MAE ($)', 'RMSE ($)', 'MAPE (%)',
    'Directional Accuracy (%)', 'N Forecasts'
]

# Sort by MAE
model_comparison = model_comparison.sort_values(['Commodity', 'MAE ($)'])

# Display as styled table
display(model_comparison.style.format({
    'MAE ($)': '{:.2f}',
    'RMSE ($)': '{:.2f}',
    'MAPE (%)': '{:.1f}',
    'Directional Accuracy (%)': '{:.1f}',
    'N Forecasts': '{:,.0f}'
}).background_gradient(subset=['MAE ($)', 'RMSE ($)'], cmap='RdYlGn_r'))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📅 Recent Forecast Performance (Last 30 Days)

# COMMAND ----------

# Filter to last 30 days
recent_date = df_eval['forecast_date'].max() - pd.Timedelta(days=30)
df_recent = df_eval[df_eval['forecast_date'] >= recent_date]

# Calculate recent metrics
recent_metrics = df_recent.groupby(['commodity', 'model_version']).agg({
    'abs_error': 'mean',
    'squared_error': lambda x: np.sqrt(x.mean()),
    'direction_correct': lambda x: x.mean() * 100,
    'forecast_date': 'count'
}).reset_index()

recent_metrics.columns = [
    'Commodity', 'Model', 'MAE ($)', 'RMSE ($)',
    'Directional Accuracy (%)', 'N Forecasts'
]

print(f"📊 Performance over last 30 days ({len(df_recent):,} forecasts)")
display(recent_metrics.style.format({
    'MAE ($)': '{:.2f}',
    'RMSE ($)': '{:.2f}',
    'Directional Accuracy (%)': '{:.1f}',
    'N Forecasts': '{:,.0f}'
}).background_gradient(subset=['MAE ($)'], cmap='RdYlGn_r'))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎲 Forecast Uncertainty (Confidence Intervals)

# COMMAND ----------

# Load distribution data if available
try:
    df_dist = spark.read.parquet('/FileStore/production_forecasts/distributions.parquet').toPandas()

    # Calculate percentiles for confidence intervals
    # (This would require unpivoting the day_1, day_2, ... columns)
    print("✅ Distribution data loaded")
    print(f"   Total paths: {len(df_dist):,}")

except Exception as e:
    print("⚠️ Distribution data not available")
    print(f"   Error: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📝 Summary & Recommendations

# COMMAND ----------

# Generate automated insights
best_model_coffee = model_comparison[model_comparison['Commodity'] == 'Coffee'].iloc[0]
best_model_sugar = model_comparison[model_comparison['Commodity'] == 'Sugar'].iloc[0]

displayHTML(f"""
<div style="background: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 4px solid #667eea;">
    <h3>🎯 Key Insights</h3>

    <h4>Best Performing Models:</h4>
    <ul>
        <li><strong>Coffee:</strong> {best_model_coffee['Model']} (MAE: ${best_model_coffee['MAE ($)']:.2f})</li>
        <li><strong>Sugar:</strong> {best_model_sugar['Model']} (MAE: ${best_model_sugar['MAE ($)']:.2f})</li>
    </ul>

    <h4>Overall Performance:</h4>
    <ul>
        <li>Average MAE across all models: ${mae:.2f}</li>
        <li>Directional accuracy: {directional_acc:.1f}% (better than random 50%)</li>
        <li>Total forecasts evaluated: {len(df_eval):,}</li>
    </ul>

    <h4>💡 Recommendations:</h4>
    <ul>
        <li>Monitor forecast accuracy weekly using this dashboard</li>
        <li>Retrain models if MAE increases by >20% from baseline</li>
        <li>Consider ensemble methods to combine top-performing models</li>
        <li>Add new exogenous variables if directional accuracy falls below 55%</li>
    </ul>
</div>
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC **Dashboard Controls:**
# MAGIC 1. Use the widgets at the top to filter by commodity, model, and forecast horizon
# MAGIC 2. Re-run all cells to refresh the dashboard with new filters
# MAGIC 3. Data is loaded from `/FileStore/production_forecasts/`
# MAGIC
# MAGIC **Next Steps:**
# MAGIC - Schedule this notebook to run daily/weekly
# MAGIC - Set up email alerts when performance degrades
# MAGIC - Add more advanced visualizations (residual analysis, autocorrelation)
