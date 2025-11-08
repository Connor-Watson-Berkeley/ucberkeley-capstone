#!/bin/bash

# Fix Unity Catalog /etc/hosts redirection
# This removes the broken PrivateLink redirect that causes UC queries to hang

echo "Fixing Unity Catalog /etc/hosts redirection..."

# Backup original
cp /etc/hosts /etc/hosts.backup

# Remove the broken oregon.cloud.databricks.com redirect
sed -i '/oregon.cloud.databricks.com/d' /etc/hosts

echo "✅ Removed broken Unity Catalog redirect from /etc/hosts"
echo "New /etc/hosts content:"
cat /etc/hosts

echo "Done"
