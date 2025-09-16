# 🔒 Key Rotation Guide

## Overview
This guide outlines the process for rotating API keys, secrets, and credentials for the trading bot.

## Key Types

### 1. Hyperliquid API Keys
- **Location**: `config/credentials/`
- **Files**: `hyperliquid_api_key.json`, `hyperliquid_secret.json`
- **Rotation Frequency**: Monthly or on security incident

### 2. Database Credentials
- **Location**: `config/credentials/database.env`
- **Rotation Frequency**: Quarterly

### 3. Webhook URLs
- **Location**: `config/webhooks.json`
- **Rotation Frequency**: On compromise

## Rotation Process

### Step 1: Generate New Keys
```bash
# Generate new Hyperliquid API key
python scripts/security/generate_new_keys.py --type hyperliquid

# Generate new database credentials
python scripts/security/generate_new_keys.py --type database
```

### Step 2: Update Configuration
```bash
# Update API keys
python scripts/security/update_credentials.py --type hyperliquid --new-key <new_key>

# Update database credentials
python scripts/security/update_credentials.py --type database --new-credentials <new_creds>
```

### Step 3: Test New Keys
```bash
# Test API connectivity
python scripts/security/test_credentials.py --type hyperliquid

# Test database connectivity
python scripts/security/test_credentials.py --type database
```

### Step 4: Deploy and Monitor
```bash
# Deploy new configuration
python scripts/deployment/deploy_config.py

# Monitor for errors
python scripts/monitoring/check_system_health.py
```

### Step 5: Revoke Old Keys
```bash
# Revoke old API keys
python scripts/security/revoke_old_keys.py --type hyperliquid --old-key <old_key>
```

## Emergency Rotation

### Immediate Key Rotation
```bash
# Emergency rotation script
python scripts/security/emergency_rotation.py --reason "security_incident"
```

### Rollback Procedure
```bash
# Rollback to previous keys
python scripts/security/rollback_keys.py --backup-file <backup_file>
```

## Security Best Practices

1. **Never commit keys to version control**
2. **Use environment variables for sensitive data**
3. **Encrypt credentials at rest**
4. **Monitor key usage and access patterns**
5. **Implement key expiration policies**
6. **Use least privilege principle**

## Monitoring

### Key Usage Monitoring
- Track API key usage patterns
- Monitor for unusual access patterns
- Alert on key expiration

### Security Alerts
- Failed authentication attempts
- Unusual API usage patterns
- Key rotation events

## Backup and Recovery

### Key Backup
```bash
# Backup current keys
python scripts/security/backup_keys.py --output backup_keys_$(date +%Y%m%d).json
```

### Key Recovery
```bash
# Restore from backup
python scripts/security/restore_keys.py --backup-file backup_keys_20240101.json
```

## Compliance

### Audit Trail
- All key rotations are logged
- Access patterns are monitored
- Compliance reports generated monthly

### Documentation
- Key rotation procedures documented
- Security incidents tracked
- Regular security reviews conducted
