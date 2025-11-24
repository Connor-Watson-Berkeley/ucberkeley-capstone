# Claude Code Configuration

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
- Documentation is in `research_agent/infrastructure/docs/` and `research_agent/infrastructure/ACTIVE_COMPONENTS.md`
- Delete obsolete files when consolidating

## TODO List Management

- Update TODO list in real-time as work progresses
- Mark tasks complete IMMEDIATELY after finishing
- Keep exactly ONE task as in_progress at a time
- Remove tasks that are no longer relevant

---

**This configuration ensures consistent behavior across context resumptions and prevents wasting time on already-completed work.**
