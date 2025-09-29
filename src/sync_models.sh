#!/bin/bash
set -eo pipefail

# Ensure environment variables are set
: "${HDFS_MODEL_REPO?HDFS_MODEL_REPO is not set}"
: "${LOCAL_MODEL_REPO?LOCAL_MODEL_REPO is not set}"

# Wait for HDFS to be ready
echo "Waiting for HDFS to exit safe mode..."
until hdfs dfsadmin -safemode get | grep -q "Safe mode is OFF"; do
  echo -n "."
  sleep 5
done
echo "HDFS is ready."

# Ensure the HDFS model repository exists
hdfs dfs -mkdir -p "$HDFS_MODEL_REPO"

# Continuous sync loop
while true; do
  echo "Checking for updates in HDFS path: ${HDFS_MODEL_REPO}"
  
  # Create a temporary directory for the download
  TMP_DIR=$(mktemp -d)

  # Attempt to copy all files from HDFS to the temporary directory
  # The || true prevents the script from exiting if the directory is empty
  if hdfs dfs -get "${HDFS_MODEL_REPO}/*" "$TMP_DIR" || true; then
    # Check if anything was actually downloaded
    if [ -n "$(ls -A "$TMP_DIR")" ]; then
      echo "Update detected. Syncing to local model repository."
      # Use rsync to atomically update the target directory.
      # --delete ensures that if a model version is removed from HDFS,
      # it's also removed from the local repository.
      rsync -a --delete "$TMP_DIR/" "$LOCAL_MODEL_REPO/"
      echo "Sync successful."
    else
      echo "No models found in HDFS source directory. No changes made."
    fi
  else
    echo "HDFS copy command failed. Will retry."
  fi

  # Clean up the temporary directory
  rm -rf "$TMP_DIR"
  
  # Wait for 30 seconds before the next check
  sleep 30
done
