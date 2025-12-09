from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

print("=" * 100)
print("CHECKING BACKTEST RESULTS TABLES")
print("=" * 100)

# List all tables matching backtest results pattern
tables = spark.sql("""
    SHOW TABLES IN commodity.trading_agent
    LIKE 'results_%_by_year_%'
""").collect()

print(f"\nFound {len(tables)} backtest results tables:")
for table_row in tables:
    table_name = table_row.tableName
    print(f"\n  • {table_name}")

    # Get row count
    count = spark.sql(f"SELECT COUNT(*) as n FROM commodity.trading_agent.{table_name}").first().n
    print(f"    {count} rows")

# Check for manifest-identified models
print("\n" + "=" * 100)
print("CHECKING FOR EXCELLENT/GOOD MODELS")
print("=" * 100)

excellent_models = [
    ('coffee', 'naive'),
    ('coffee', 'xgboost'),
]

marginal_models = [
    ('coffee', 'sarimax_auto_weather'),
    ('sugar', 'naive'),
]

print("\nEXCELLENT models:")
for commodity, model in excellent_models:
    table_name = f"results_{commodity}_by_year_{model}"
    exists = spark.sql(f"""
        SHOW TABLES IN commodity.trading_agent
        LIKE '{table_name}'
    """).count() > 0

    if exists:
        count = spark.sql(f"SELECT COUNT(*) as n FROM commodity.trading_agent.{table_name}").first().n
        print(f"  ✓ {commodity}/{model}: {count} rows")
    else:
        print(f"  ✗ {commodity}/{model}: NO BACKTEST TABLE")

print("\nMARGINAL models:")
for commodity, model in marginal_models:
    table_name = f"results_{commodity}_by_year_{model}"
    exists = spark.sql(f"""
        SHOW TABLES IN commodity.trading_agent
        LIKE '{table_name}'
    """).count() > 0

    if exists:
        count = spark.sql(f"SELECT COUNT(*) as n FROM commodity.trading_agent.{table_name}").first().n
        print(f"  ✓ {commodity}/{model}: {count} rows")
    else:
        print(f"  ✗ {commodity}/{model}: NO BACKTEST TABLE")

print("\n" + "=" * 100)
