"""
Wait for a Databricks cluster to reach RUNNING state
"""
import subprocess
import json
import time
import sys

CLUSTER_ID = "1107-063604-wgjrug42"
MAX_WAIT_SECONDS = 600  # 10 minutes
CHECK_INTERVAL = 20  # seconds

def get_cluster_state():
    """Get current cluster state via databricks CLI"""
    try:
        result = subprocess.run(
            ["databricks", "clusters", "get", "--cluster-id", CLUSTER_ID],
            capture_output=True,
            text=True,
            check=True
        )
        cluster_info = json.loads(result.stdout)
        return cluster_info.get("state"), cluster_info.get("state_message", "")
    except Exception as e:
        print(f"❌ Error getting cluster state: {e}")
        return None, str(e)

def main():
    print(f"Waiting for cluster {CLUSTER_ID} to start...")
    print(f"Max wait time: {MAX_WAIT_SECONDS}s, checking every {CHECK_INTERVAL}s")
    print("")

    waited = 0
    iteration = 1

    while waited < MAX_WAIT_SECONDS:
        state, message = get_cluster_state()

        print(f"  [{iteration}] State: {state}")
        if message and state != "RUNNING":
            print(f"      Message: {message}")

        if state == "RUNNING":
            print("")
            print("✅ Cluster is RUNNING!")
            return True
        elif state in ["ERROR", "TERMINATED", "TERMINATING"]:
            print("")
            print(f"❌ Cluster failed to start: {state}")
            print(f"   Message: {message}")
            return False

        time.sleep(CHECK_INTERVAL)
        waited += CHECK_INTERVAL
        iteration += 1

    print("")
    print(f"⚠️  Cluster still not running after {MAX_WAIT_SECONDS}s")
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
