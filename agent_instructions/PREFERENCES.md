# Connor's Development Preferences

**Critical: Read this first when starting any AI session on this project**

## Code Style

- **Concise over verbose**: Keep code and documentation tight
- **PySpark over pandas**: Maximize parallelization, minimize pandas usage
- **Start simple, layer complexity**: Don't over-engineer upfront
- **Scalable from day one**: Build for growth but implement incrementally

## Documentation Philosophy

- **Concise everywhere except agent_instructions/**: Keep docs brief
- **Detailed only for AI context**: agent_instructions/ can be verbose for AI understanding
- **Code comments**: Minimal, self-documenting code preferred
- **No unnecessary markdown files**: Only create docs when explicitly needed

## Databricks Workflow

- **Python modules first**: Write .py files, not notebooks for core logic
- **Notebooks for**: Visualization, experimentation, evaluation
- **Local development**: Use local Parquet snapshots for testing
- **Dual-mode code**: Functions should work locally and in Databricks

## Testing Strategy

- **Tonight's goal**: Get models training in Databricks for evaluation
- **Start with baseline**: Fitted ARIMA on close price timeseries
- **Iterate fast**: Train multiple models in parallel, evaluate later

## Communication Style

- **Direct and concise**: No fluff
- **Show file:line references**: Use `file.py:123` format
- **Prioritize action**: Less planning, more building

## Project Context Awareness

- **Week 10-11 of 12**: Time pressure, focus on deliverables
- **Role**: Time Series Models Lead (Agent T - Forecaster)
- **Dependencies**: Research Agent (Francisco) provides unified_data
- **Consumer**: Risk/Trading Agent uses forecast outputs

## Key Priorities (in order)

1. Get baseline models training tonight
2. Build scalable model bank framework
3. Train & evaluate multiple models
4. Deliver clean forecasts to trading agent
5. Document what matters

## AI Collaboration

- **When to ask**: Architectural decisions, multiple valid approaches
- **When to proceed**: Implementation details, obvious next steps
- **Parallelize**: Run multiple tool calls when possible
- **Use todos**: Track multi-step tasks always
