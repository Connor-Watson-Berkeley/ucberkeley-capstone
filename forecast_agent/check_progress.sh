#!/bin/bash
set -a && source ../infra/.env && set +a

while true; do
    echo ""
    echo "========== $(date) =========="
    echo ""
    
    # Show latest log entries
    if [ -f autonomous_monitor.log ]; then
        echo "Latest autonomous monitor activity:"
        tail -20 autonomous_monitor.log
    fi
    
    # Show backfill status
    echo ""
    echo "Active backfill processes:"
    ps aux | grep "backfill_rolling_window" | grep -v grep || echo "  None"
    
    # Sleep for 15 minutes
    sleep 900
done
