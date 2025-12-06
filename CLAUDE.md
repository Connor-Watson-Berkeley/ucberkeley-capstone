# Claude Code Workflow Instructions

**Purpose:** Checklist to prevent errors when working across this multi-component capstone project

---

## Documentation Structure (CRITICAL)

**This project uses hierarchical documentation.** Before performing ANY task:

1. **Read relevant documentation FIRST** from the appropriate folder
2. Start with the component README.md for overview
3. Follow links to detailed docs/ files for specifics
4. Never search for files - all paths are explicit in the hierarchy

**Full Documentation Strategy**: See [docs/DOCUMENTATION_STRATEGY.md](docs/DOCUMENTATION_STRATEGY.md) for:
- Complete hierarchical structure explanation
- "Read X before doing Y" pattern
- Temp document lifecycle and cleanup
- Reference rules and best practices

**Example workflow:**
- Before forecasting work → Read [forecast_agent/README.md](forecast_agent/README.md), then follow links to docs/
- Before research work → Read [research_agent/README.md](research_agent/README.md) for navigation
- Before any task → Check [docs/DOCUMENTATION_STRATEGY.md](docs/DOCUMENTATION_STRATEGY.md) if unsure about doc organization

**Forecast Agent - Read X Before Doing Y:**
- **Before training models** → Read [forecast_agent/docs/ARCHITECTURE.md](forecast_agent/docs/ARCHITECTURE.md) sections on "Train-Once Pattern" and "Model Persistence"
- **Before running Spark backfills** → Read [forecast_agent/docs/SPARK_BACKFILL_GUIDE.md](forecast_agent/docs/SPARK_BACKFILL_GUIDE.md) for cluster sizing and cost optimization
- **Before modifying models** → Read [forecast_agent/docs/ARCHITECTURE.md](forecast_agent/docs/ARCHITECTURE.md) section on "Model Implementation Pattern"
- **Before large backfills** → Read [forecast_agent/README.md](forecast_agent/README.md) for execution environment guidance (local vs Databricks)

---

## Before Making Any Code Changes

### 1. Read Documentation in Your Working Folder (REQUIRED)
Before writing code in ANY component, ALWAYS read the README.md in that folder:

```bash
# Working in forecast_agent/?
cat forecast_agent/README.md

# Working in research_agent/?
cat research_agent/README.md

# Working in trading_agent/?
cat trading_agent/README.md
```

**Rule:** Read the local README first. It will point you to other docs you need.

### 2. Follow Cross-References (Hierarchical Navigation)
READMEs use **hierarchical documentation** - each component has a docs/ folder with detailed guides:

```
forecast_agent/README.md (concise overview)
  ↓ links to
forecast_agent/docs/ARCHITECTURE.md  (detailed implementation)
  ↓ references
research_agent/docs/UNIFIED_DATA_ARCHITECTURE.md  (data source authority)
```

**Key docs by topic:**
- **Documentation strategy:** `docs/DOCUMENTATION_STRATEGY.md` (read this to understand our doc organization)
- **Data architecture:** `research_agent/docs/UNIFIED_DATA_ARCHITECTURE.md`
- **Forecasting architecture:** `forecast_agent/docs/ARCHITECTURE.md`
- **Spark parallelization:** `forecast_agent/docs/SPARK_BACKFILL_GUIDE.md`
- **Data sources:** `research_agent/docs/DATA_SOURCES.md`
- **Trading system:** `trading_agent/README.md`

**IMPORTANT**: All documentation is reachable from root README.md through explicit links. Never search for files - follow the hierarchy.

### 3. Data Source Rule (Forecasting Only)
When writing **forecasting code specifically**:

❌ **DON'T** query `commodity.bronze.*` tables
✅ **DO** query `commodity.silver.unified_data`

**Why:**
- unified_data has continuous daily coverage (including weekends/holidays)
- All features are forward-filled (no NULLs)
- Bronze tables have gaps (trading days only)

**Note:** Bronze tables are fine for other use cases (data exploration, debugging, etc.)

### 4. Check for Existing Patterns
Before implementing new functionality:

```bash
# Search for similar implementations
grep -r "pattern_name" --include="*.py"
```

**Example:** Before adding a new model:
1. Read [forecast_agent/docs/ARCHITECTURE.md](forecast_agent/docs/ARCHITECTURE.md) section on "Model Implementation Pattern"
2. Check existing models in `forecast_agent/ground_truth/models/`
3. Follow the train/predict separation pattern

---

## Data Architecture Quick Reference

```
Bronze (Raw)
  └── commodity.bronze.market          # Trading days only, has gaps
  └── commodity.bronze.weather         # Daily, complete
  └── commodity.bronze.vix             # Trading days only
  └── commodity.bronze.forex           # Weekdays only
         ↓
    Forward-fill to continuous daily
         ↓
Silver (Unified)
  └── commodity.silver.unified_data    # ⚠️ USE THIS FOR FORECASTING
      - Grain: (date, commodity, region)
      - Coverage: Every day since 2015-07-07
      - Forward-filled: No NULLs
      - Trading flag: is_trading_day column
         ↓
Gold (Forecasts)
  └── commodity.forecast.distributions # Model outputs
```

**Golden Rule:** All forecasting models should query `unified_data`, not bronze tables.

---

## Common Pitfalls (Learn from Past Mistakes)

### ❌ Mistake #1: Querying bronze.market Instead of unified_data
**What happened:** TFT implementation queried `bronze.market` which only has trading days, causing "missing timesteps" error.

**Why wrong:** Bronze tables have gaps (weekends/holidays missing).

**Correct approach:** Query `unified_data` which has continuous daily data with forward-filled prices.

**File reference:** `research_agent/docs/UNIFIED_DATA_ARCHITECTURE.md` lines 266-276

### ❌ Mistake #2: Creating Docs Without Being Asked
**What happened:** Created `TFT_STATUS.md` proactively without user request.

**Why wrong:** User's instructions say "NEVER proactively create documentation files (*.md)".

**Correct approach:** Only create docs when explicitly requested.

### ❌ Mistake #3: Not Checking git Before Committing
**What happened:** Almost committed hardcoded Databricks credentials in 3 files.

**Why wrong:** GitHub secret scanning would block the push.

**Correct approach:**
```bash
git diff                    # Review all changes
grep -r "dapi" --include="*.py"  # Check for hardcoded tokens
```

---

## File Permissions / Ownership

### ✅ You Can Modify
- `research_agent/*` (data pipelines)
- `forecast_agent/*` (your forecasting models)
- `collaboration/*` (shared docs)
- `docs/*` (architecture docs)

### ⚠️ Ask First
- `infra/*` (infrastructure changes)
- Root-level config files

### ❌ Don't Touch
- `trading_agent/*` (Tony's code)
- `.env` files (credentials)

---

## Credential Management

### ✅ Correct Pattern
```python
import os
token = os.environ['DATABRICKS_TOKEN']
```

### ❌ Wrong Pattern
```python
token = "dapi_fake_example_token_12345"  # Hardcoded! Never do this!
```

**Always use:** Environment variables via `os.environ` or load from `../infra/.env`

---

## Before Pushing to Git

### Pre-Push Checklist
```bash
# 1. Review all changes
git status
git diff

# 2. Check for hardcoded secrets
grep -r "dapi" forecast_agent/ research_agent/
grep -r "https://dbc-" forecast_agent/ research_agent/

# 3. Verify no trading_agent changes (unless explicitly asked)
git status | grep trading_agent

# 4. Test locally first
python -m pytest tests/
```

### Git Commit Message Format
```
Brief description (imperative mood)

- Bullet points of what changed
- Why the change was needed

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

---

## Quick Wins

### Instead of Guessing, Check Docs
```bash
# Before: "I think I should query bronze.market"
# After: "Let me read UNIFIED_DATA_ARCHITECTURE.md first"

cat research_agent/docs/UNIFIED_DATA_ARCHITECTURE.md | grep -A 10 "unified_data"
```

### Instead of Creating Temp Files, Ask
```bash
# Before: Write TFT_STATUS.md
# After: "Should I document this?"
```

### Instead of Assuming, Verify
```bash
# Before: "Coffee data has weekends"
# After: Query unified_data to check date coverage
```

---

## Workflow Summary

```
┌─────────────────────────────────────────┐
│ User Requests Feature                   │
└───────────┬─────────────────────────────┘
            ↓
┌─────────────────────────────────────────┐
│ 1. Read relevant docs FIRST             │
│    - Component README.md for overview   │
│    - Follow links to docs/ for details  │
│    - Check DOCUMENTATION_STRATEGY.md    │
└───────────┬─────────────────────────────┘
            ↓
┌─────────────────────────────────────────┐
│ 2. Understand data source               │
│    - Use unified_data for forecasting   │
│    - Check grain, coverage, nulls       │
└───────────┬─────────────────────────────┘
            ↓
┌─────────────────────────────────────────┐
│ 3. Implement solution                   │
│    - Follow existing patterns           │
│    - Use env vars for credentials       │
└───────────┬─────────────────────────────┘
            ↓
┌─────────────────────────────────────────┐
│ 4. Test locally                         │
│    - Query Databricks to verify         │
│    - Check for edge cases               │
└───────────┬─────────────────────────────┘
            ↓
┌─────────────────────────────────────────┐
│ 5. Review before commit                 │
│    - git diff                           │
│    - Check for secrets                  │
│    - Verify no trading_agent changes    │
└───────────┬─────────────────────────────┘
            ↓
┌─────────────────────────────────────────┐
│ 6. Commit and push                      │
└─────────────────────────────────────────┘
```

---

## Emergency Reference

**If in doubt:**
1. Read [docs/DOCUMENTATION_STRATEGY.md](docs/DOCUMENTATION_STRATEGY.md) to understand doc organization
2. Read component README.md, then follow links to detailed docs/
3. Ask the user before creating new files/docs
4. Query `commodity.silver.unified_data` for forecasting
5. Never hardcode credentials
6. Don't touch `trading_agent/`

**When stuck:**
1. Read relevant documentation FIRST (follow hierarchical links)
2. Check existing code for patterns
3. Ask user for clarification
4. Don't guess - verify with data queries

**Documentation Quick Links:**
- [docs/DOCUMENTATION_STRATEGY.md](docs/DOCUMENTATION_STRATEGY.md) - How we organize docs
- [forecast_agent/README.md](forecast_agent/README.md) - Forecast agent guide
- [forecast_agent/docs/ARCHITECTURE.md](forecast_agent/docs/ARCHITECTURE.md) - Train-once architecture
- [forecast_agent/docs/SPARK_BACKFILL_GUIDE.md](forecast_agent/docs/SPARK_BACKFILL_GUIDE.md) - Spark parallelization
- [research_agent/README.md](research_agent/README.md) - Research agent guide
- [research_agent/docs/UNIFIED_DATA_ARCHITECTURE.md](research_agent/docs/UNIFIED_DATA_ARCHITECTURE.md) - Data architecture

---

**Document Owner:** Claude Code (AI Assistant)
**Last Updated:** 2025-11-12
**Purpose:** Prevent repeated mistakes, establish workflow discipline
