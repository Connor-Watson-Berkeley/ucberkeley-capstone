# Claude Code Configuration

## Auto-Compress Resumption Behavior

**CRITICAL:** After every auto-compress/context resumption, ALWAYS perform these steps IN ORDER before taking any action:

### 1. Read Master Status File (REQUIRED)
```
Read /tmp/GDELT_PROJECT_STATUS.md
```

This file contains:
- Current system status
- Architecture overview
- DynamoDB schema
- Lambda functions and deployment status
- Step Function definitions
- Recovery/retry patterns
- Important decisions and context

### 2. Review TODO List (REQUIRED)
Check the current TODO list to understand:
- What's completed vs pending
- What the current task is
- What needs to be done next

### 3. Check Background Scripts (REQUIRED)
```bash
ps aux | grep python | grep -v grep
```
Look for any running background scripts that may have completed work already.

### 4. DO NOT Create New Documentation Files
**IMPORTANT:** Update the existing `/tmp/GDELT_PROJECT_STATUS.md` file instead of creating new documentation files.

If system state changes significantly, update the master status file with new information. Do NOT proliferate new .md files.

### 5. Verify Current State Before Acting
- Check AWS resources (Lambda, SQS, DynamoDB)
- Verify what's already deployed
- Confirm what still needs to be done
- Don't duplicate work that's already complete

## AWS Credentials

**ALWAYS use boto3 with default credentials:**
```python
import boto3
client = boto3.client('lambda', region_name='us-west-2')
```

**DO NOT use AWS CLI with SSO credentials** - they don't work in this environment.

## File Management

- **ALWAYS** update existing files instead of creating new ones
- **ALWAYS** read files before editing them
- Keep the master status file `/tmp/GDELT_PROJECT_STATUS.md` up to date
- Delete obsolete files when consolidating

## TODO List Management

- Update TODO list in real-time as work progresses
- Mark tasks complete IMMEDIATELY after finishing
- Keep exactly ONE task as in_progress at a time
- Remove tasks that are no longer relevant

---

**This configuration ensures consistent behavior across context resumptions and prevents wasting time on already-completed work.**
