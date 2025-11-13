# Claude Code Workflow Instructions

**Purpose:** Checklist to prevent errors when working across this multi-component capstone project

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

### 2. Follow Cross-References
READMEs use **federated documentation** - each component owns its domain's docs:

```
forecast_agent/README.md
  ↓ references
research_agent/UNIFIED_DATA_ARCHITECTURE.md  (data source authority)
  ↓ references
research_agent/DATA_SOURCES.md  (raw data sources)
```

**Key docs by topic:**
- **Data architecture:** `research_agent/UNIFIED_DATA_ARCHITECTURE.md`
- **Forecasting models:** `forecast_agent/README.md`
- **Data sources:** `research_agent/DATA_SOURCES.md`
- **Trading system:** `trading_agent/README.md`

### 3. Data Source Rule (Forecasting Only)
When writing **forecasting code specifically**:

❌ **DON'T** query `commodity.bronze.*` tables
✅ **DO** query `commodity.silver.unified_data`

**Why:**
- unified_data has continuous daily coverage (including weekends/holidays)
- All features are forward-filled (no NULLs)
- Bronze tables have gaps (trading days only)

**Note:** Bronze tables are fine for other use cases (data exploration, debugging, etc.)

### 3. Check for Existing Patterns
Before implementing new functionality:

```bash
# Search for similar implementations
grep -r "pattern_name" --include="*.py"
```

**Example:** Before adding a new model, check existing models in `forecast_agent/ground_truth/models/`

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

**File reference:** `research_agent/UNIFIED_DATA_ARCHITECTURE.md` lines 266-276

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

cat research_agent/UNIFIED_DATA_ARCHITECTURE.md | grep -A 10 "unified_data"
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
│ 1. Read relevant docs                   │
│    - UNIFIED_DATA_ARCHITECTURE.md       │
│    - Component README.md                │
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
1. Read `research_agent/UNIFIED_DATA_ARCHITECTURE.md`
2. Ask the user before creating new files/docs
3. Query `commodity.silver.unified_data` for forecasting
4. Never hardcode credentials
5. Don't touch `trading_agent/`

**When stuck:**
1. Check existing code for patterns
2. Read component README
3. Ask user for clarification
4. Don't guess - verify with data queries

---

**Document Owner:** Claude Code (AI Assistant)
**Last Updated:** 2025-11-12
**Purpose:** Prevent repeated mistakes, establish workflow discipline
