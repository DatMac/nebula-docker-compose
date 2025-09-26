#!/bin/bash
set -e

# 1. Set up Hadoop environment
export HADOOP_CONF_DIR=${HADOOP_CONF_DIR:-/opt/hadoop-2.7.4/etc/hadoop}
export CLASSPATH=$(${HADOOP_HOME}/bin/hadoop classpath --glob)


# 2. Robust Rank Assignment using HDFS as a Coordinator
# Workers will race to create a directory for each rank. The winner gets the rank.
COORD_PATH="${HDFS_COORD_PATH:-/job-coordination}" # A shared directory on HDFS
export NUM_WORKERS=${NUM_WORKERS:-1} # Total number of workers
MAX_WORKERS=$((NUM_WORKERS - 1))
ASSIGNED_RANK=-1

echo "Attempting to acquire a rank from 0 to ${MAX_WORKERS}..."
hdfs dfs -mkdir -p "$COORD_PATH"

for rank in $(seq 0 $MAX_WORKERS); do
  # Try to atomically create a directory for this rank
  if hdfs dfs -mkdir "${COORD_PATH}/rank_${rank}" >/dev/null 2>&1; then
    # Success! We claimed this rank.
    ASSIGNED_RANK=$rank
    echo "Successfully acquired WORKER_RANK: ${ASSIGNED_RANK}"
    
    # Optional: Write our identity to the directory for debugging
    hostname -i > "/tmp/my_ip"
    hdfs dfs -put -f "/tmp/my_ip" "${COORD_PATH}/rank_${rank}/"
    break
  else
    # This rank was already taken.
    echo "Rank ${rank} is already taken. Trying next."
    sleep 1 # Wait a moment to avoid a thundering herd
  fi
done


# 3. Final Check
if [ "$ASSIGNED_RANK" -eq -1 ]; then
  echo "ERROR: Could not acquire a worker rank. All ranks may be taken."
  exit 1
fi

export WORKER_RANK=${ASSIGNED_RANK}
export RANK=${ASSIGNED_RANK}
export WORLD_SIZE=${NUM_WORKERS}

# ==================== NEW SOLUTION FROM DOCS ==================================
echo "[Rank ${RANK}] Discovering network interface for MASTER_ADDR=${MASTER_ADDR}"

# The MASTER_ADDR is the hostname of the rank 0 container. We need its IP address.
# `getent hosts` is a robust way to perform DNS lookup inside the container.
MASTER_IP=$(getent hosts "$MASTER_ADDR" | awk '{ print $1 }')

if [ -z "$MASTER_IP" ]; then
    echo "ERROR: Could not resolve MASTER_ADDR '$MASTER_ADDR'. Exiting."
    exit 1
fi
echo "[Rank ${RANK}] Resolved MASTER_ADDR to ${MASTER_IP}"

if [ "$RANK" -eq 0 ]; then
  # This is the master node. It needs to find the interface that has its own IP.
  INTERFACE=$(ip -o -4 addr show | awk -v ip="$MASTER_IP" '$4 ~ ip {print $2}')
else
  # These are worker nodes. They need to find the interface used to route to the master.
  INTERFACE=$(ip route get "$MASTER_IP" | grep -oP '(?<=dev )[^ ]+')
fi

if [ -z "$INTERFACE" ]; then
    echo "ERROR: Could not discover a network interface. Exiting."
    exit 1
fi

echo "[Rank ${RANK}] Discovered network interface: ${INTERFACE}. Exporting..."
export GLOO_SOCKET_IFNAME=$INTERFACE
export TP_SOCKET_IFNAME=$INTERFACE
# ==============================================================================

# 4. Dynamically construct data paths based on the assigned rank
HDFS_PARTITION_PATH="${HDFS_DATA_PATH}/part_${WORKER_RANK}"
LOCAL_PARTITION_PATH="${LOCAL_DATA_PATH}/part_${WORKER_RANK}"


# 5. Copy the specific data partition from HDFS
if [ -n "$HDFS_DATA_PATH" ]; then
  mkdir -p "$LOCAL_PARTITION_PATH"
  echo "Copying shared metadata files from HDFS path: '${HDFS_DATA_PATH}'..."
  hdfs dfs -get "${HDFS_DATA_PATH}/META.json" "${LOCAL_DATA_PATH}/"
  hdfs dfs -get "${HDFS_DATA_PATH}/node_map.pt" "${LOCAL_DATA_PATH}/"
  hdfs dfs -get "${HDFS_DATA_PATH}/edge_map.pt" "${LOCAL_DATA_PATH}/"
  hdfs dfs -get "${HDFS_DATA_PATH}/labels.pt" "${LOCAL_DATA_PATH}/"

  #mkdir -p "${LOCAL_DATA_PATH}/part_1/"
  #mkdir -p "${LOCAL_DATA_PATH}/part_2/"
  #hdfs dfs -get "${HDFS_DATA_PATH}/part_1/*" "${LOCAL_DATA_PATH}/part_1"
  #hdfs dfs -get "${HDFS_DATA_PATH}/part_2/*" "${LOCAL_DATA_PATH}/part_2"

  echo "Copying data for rank ${WORKER_RANK} from '${HDFS_PARTITION_PATH}' to '${LOCAL_PARTITION_PATH}'..."
  if hdfs dfs -test -e "$HDFS_PARTITION_PATH"; then
    hdfs dfs -get "$HDFS_PARTITION_PATH"/* "$LOCAL_PARTITION_PATH"
    echo "Copy complete."
  else
    echo "ERROR: HDFS source path '$HDFS_PARTITION_PATH' does not exist."
    exit 1
  fi
else
  echo "HDFS_DATA_PATH not set, skipping data copy."
fi

# 6. Execute the main command passed to the container
echo "Executing main command: $@"
exec "$@"
