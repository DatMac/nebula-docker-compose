from __future__ import annotations

import logging
import subprocess

import pendulum
from docker.types import Mount

from airflow.models.param import Param
from airflow.exceptions import AirflowException
from airflow.models.dag import DAG, DagParam
from airflow.operators.python import BranchPythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.providers.docker.operators.docker import DockerOperator

# Set up logging
log = logging.getLogger(__name__)

# --- Constants and Helper Functions from nebula_data_importer_dag ---

# Constants for nebula-importer
HOST_IMPORTER_CONFIG_PATH = "/home/macquangdat2412/nebula-docker-compose/nebula_importer/config/hdfs_config.yaml"
CONTAINER_CONFIG_PATH = "/configs/hdfs_config.yaml"
HOST_LOG_PATH = "/home/macquangdat2412/nebula-docker-compose/nebula_importer/logs"


def find_file_and_get_branch_task(**kwargs):
    """
    Lists files in an HDFS directory, finds the first valid data file,
    and returns the appropriate downstream task_id based on its extension.
    """
    folder_path = kwargs["params"]["hdfs_folder_path"]
    log.info(f"Scanning HDFS folder: {folder_path}")

    # Use 'hdfs dfs -ls -C' to get a clean list of file paths in the directory
    bash_command = f"hdfs dfs -ls -R {folder_path}"

    try:
        # Execute the HDFS command
        result = subprocess.check_output(bash_command, shell=True, text=True)
        all_files = result.strip().split("\n")

        # Filter out common non-data files (e.g., _SUCCESS, hidden files)
        data_files = [
            f
            for f in all_files
            if not f.split("/")[-1].startswith(("_", ".")) and f.strip()
        ]

        if not data_files:
            raise AirflowException(f"No valid data files found in HDFS folder: {folder_path}")

        # Select the first valid data file found
        file_to_check = data_files[-1]
        log.info(f"Found data file to check: {file_to_check}")

        if file_to_check.lower().endswith(".csv"):
            log.info("File is a CSV. Routing to nebula_importer_task.")
            return "nebula_importer_task"
        else:
            log.info("File is not a CSV. Routing to submit_nebula_exchange_job.")
            return "submit_nebula_exchange_job"

    except subprocess.CalledProcessError as e:
        log.error(f"Failed to list files in HDFS. Error: {e}")
        raise AirflowException(f"HDFS command failed. Ensure the path '{folder_path}' is correct and accessible.")


# --- Constants from gcn_data_preparation_dag ---

SPARK_PACKAGES_GCN = [
    "com.vesoft.nebula-spark-connector:nebula-spark-connector_2.2:3.4.0",
    "com.datastax.spark:spark-cassandra-connector_2.11:2.4.3",
    "graphframes:graphframes:0.8.1-spark2.4-s_2.11",
    "com.twitter:jsr166e:1.1.0"
]

# --- Constants from pytorch_distributed_training_dag ---

torch_and_hdfs_command = """
bash -c "
# --- Part 1: Training ---
torchrun \\
--nproc_per_node=3 \\
--nnodes=1 \\
--rdzv_backend=c10d \\
--rdzv_endpoint=localhost:29500 \\
/app/distributed_training.py \\
--dataset_root_dir /pyg_dataset \\
--num_epochs 10 \\
--batch_size 512 \\
--num_neighbors '10,5' \\
--progress_bar \\
--num_workers=0 \\
--model_save_path /app/my_models \\
&& \\
echo '--- Training finished, uploading model checkpoint to HDFS ---' && \\
hdfs dfs -mkdir -p /models/{{ ts_nodash }} && \\
hdfs dfs -put /app/my_models/* /models/{{ ts_nodash }} && \\
echo '--- Upload complete ---' && \\
echo '--- Exporting latest model for Triton Inference Server ---' && \\
python3 /app/export_for_triton.py \\
--model_name graphsage_model \\
--checkpoint_path /app/my_models/model_epoch_10.pt \\
--dataset_meta_path /pyg_dataset/META.json \\
--hdfs_repo_path /triton_models
"
"""

# --- Unified DAG Definition ---

with DAG(
    dag_id="ml_pipeline_dag",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    schedule="@daily",
    catchup=False,
    tags=["ml-pipeline", "nebula", "spark", "pytorch", "unified"],
    params={
        "hdfs_folder_path": Param(  # <-- CORRECT CLASS
            default="/telecom/vertices/Customer",
            type="string",  # <-- 'type' is now valid
            title="HDFS Folder Path",
            description="The HDFS directory containing the data file to import.",
        )
    }, 
    doc_md="""
    ### Unified End-to-End Machine Learning Pipeline

    This DAG orchestrates a full machine learning workflow, from data ingestion into NebulaGraph to distributed training of a PyTorch model.

    **Execution Order and Workflow:**

    1.  **`nebula_data_importer`**:
        - A branching task inspects a specified HDFS folder.
        - If a `.csv` file is found, it uses the `nebula-importer` (Docker) to load data.
        - Otherwise, it submits a `nebula-exchange` Spark job.

    2.  **`incremental_materialize`**:
        - A Spark job that incrementally processes data from HDFS and loads it into Cassandra, based on the daily schedule.

    3.  **`gcn_data_preparation`**:
        - A Spark job that prepares graph data for GNN training by:
            - Loading graph data from NebulaGraph.
            - Enriching features from Cassandra.
            - Partitioning the graph using GraphFrames.
            - Saving the prepared data to HDFS.

    4.  **`pytorch_distributed_training`**:
        - A Docker container is launched to run a distributed PyTorch training job using `torchrun`.
        - The trained model artifacts are then uploaded to HDFS.
    """,
) as dag:
    # --- Tasks from nebula_data_importer_dag ---
    find_and_branch_task = BranchPythonOperator(
        task_id="find_file_and_branch",
        python_callable=find_file_and_get_branch_task,
    )

    nebula_importer_task = DockerOperator(
        task_id="nebula_importer_task",
        image="vesoft/nebula-importer:v4",
        docker_conn_id=None,
        command=f"--config {CONTAINER_CONFIG_PATH}",
        network_mode="nebula-docker-compose_nebula-net",
        mounts=[
            Mount(
                source=HOST_IMPORTER_CONFIG_PATH,
                target=CONTAINER_CONFIG_PATH,
                type="bind",
                read_only=True,
            ),
            Mount(source=HOST_LOG_PATH, target="/configs/logs", type="bind"),
        ],
        auto_remove=True,
        tty=True,
    )

    submit_nebula_exchange_job = SparkSubmitOperator(
        task_id="submit_nebula_exchange_job",
        conn_id="spark-spark",
        application="/opt/spark_apps/nebula-exchange_spark_2.4-3.8.0.jar",
        java_class="com.vesoft.nebula.exchange.Exchange",
        application_args=["--config", "/opt/spark_apps/telecom_nebula_exchange.conf"],
        verbose=True,
    )

    # --- Task from incremental_materialize_dag ---
    submit_incremental_spark_job = SparkSubmitOperator(
        task_id="submit_incremental_hdfs_to_cassandra_job",
        conn_id="spark-spark",
        application="/opt/spark_apps/incremental_materialize.py",
        name="IncrementalHdfsToCassandra",
        application_args=[
            "{{ data_interval_start }}",
            "{{ data_interval_end }}",
        ],
        conf={
            "spark.jars.packages": "com.datastax.spark:spark-cassandra-connector_2.11:2.4.3,com.twitter:jsr166e:1.1.0"
        },
        packages="com.datastax.spark:spark-cassandra-connector_2.11:2.4.3,com.twitter:jsr166e:1.1.0",
        verbose=True,
        # This task should run after either of the data import tasks completes.
        trigger_rule="one_success",
    )

    # --- Task from gcn_data_preparation_dag ---
    submit_gnn_prep_job = SparkSubmitOperator(
        task_id="submit_gnn_data_prep_job",
        conn_id="spark-spark",
        application="/opt/spark_apps/data_partitioner.py",
        name="GNN_Data_Prep_Pipeline",
        packages=",".join(SPARK_PACKAGES_GCN),
        conf={
            "spark.cassandra.connection.host": "cassandra",
            "spark.cassandra.connection.port": "9042",
            "spark.sql.shuffle.partitions": "200",
        },
        verbose=True,
    )

    # --- Task from pytorch_distributed_training_dag ---
    run_pytorch_training = DockerOperator(
        task_id="run_pytorch_training_in_container",
        image="pyg-node",
        docker_conn_id=None,
        command=torch_and_hdfs_command,
        auto_remove="force",
        network_mode="nebula-docker-compose_nebula-net",
        environment={
            "JAVA_HOME": "/usr/lib/jvm/java-8-openjdk-amd64",
            "HADOOP_CONF_DIR": "/opt/hadoop-2.7.4/etc/hadoop",
            "HDFS_DATA_PATH": "/tmp/pyg_dataset",
            "LOCAL_DATA_PATH": "/pyg_dataset",
            "NUM_WORKERS": "3",
            "HDFS_COORD_PATH": "/job-coordination",
        },
        mounts=[
            Mount(
                source="/home/macquangdat2412/nebula-docker-compose/src",
                target="/app",
                type="bind"
            ),
            Mount(
                source="/home/macquangdat2412/nebula-docker-compose/hdfs-config/core-site.xml",
                target="/opt/hadoop-2.7.4/etc/hadoop/core-site.xml",
                type="bind"
            ),
            Mount(
                source="/home/macquangdat2412/nebula-docker-compose/hdfs-config/hdfs-site.xml",
                target="/opt/hadoop-2.7.4/etc/hadoop/hdfs-site.xml",
                type="bind"
            ),
        ],
        tty=True,
    )

    # --- Define Task Dependencies ---
    find_and_branch_task >> [nebula_importer_task, submit_nebula_exchange_job]
    [nebula_importer_task, submit_nebula_exchange_job] >> submit_incremental_spark_job
    submit_incremental_spark_job >> submit_gnn_prep_job
    submit_gnn_prep_job >> run_pytorch_training

