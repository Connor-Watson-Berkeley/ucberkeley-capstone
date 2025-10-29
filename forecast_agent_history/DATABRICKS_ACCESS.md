# Databricks Connect Access Issue

## Current Status
Access tokens are disabled for the organization.

**Error message**:
```
Tokens are disabled for your organization or you do not have permissions to use them.
Please contact your administrator for more information.
```

## Workspace URL
https://dbc-5474a94c-61c9.cloud.databricks.com/

## Action Items
1. Contact Databricks admin to request:
   - Personal access token generation permission
   - OR alternative auth method for Databricks Connect
2. Verify organization's security policy allows tokens

## Workaround (Current)
- Test feature engineering functions locally with pandas
- Upload code to Databricks and run in notebook for PySpark validation
- Continue development while waiting for access token

## Future Setup (Once token available)
Create `~/.databrickscfg`:
```
[DEFAULT]
host = https://dbc-5474a94c-61c9.cloud.databricks.com
token = <your-access-token>
```

Then Databricks Connect will work from local machine.
