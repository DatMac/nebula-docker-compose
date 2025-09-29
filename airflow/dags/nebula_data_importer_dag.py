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

# Constants for nebula-importer
HOST_IMPORTER_CONFIG_PATH = "/home/macquangdat2412/nebula-docker-compose/nebula_importer/config/hdfs_config.yaml"
CONTAINER_CONFIG_PATH = "/configs/hdfs_config.yaml"
HOST_LOG_PATH = "/home/macquangdat2412/nebula-docker-compose/nebula_importer/logs"
DEFAULT_HDFS_FOLDER = "/telecom/vertices/Customer"

def find_file_and_get_branch_task(**kwargs):
    """
    Lists files in an HDFS directory, finds the first valid data file,
    and returns the appropriate downstream task_id based on its extension.
    """
    folder_path = kwargs["params"].get("hdfs_folder_path", DEFAULT_HDFS_FOLDER)
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


with DAG(
    dag_id="nebula_data_importer_dag",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    schedule=None,
    catchup=False,
    tags=["nebula", "docker", "spark", "hdfs", "conditional"],
    params={
        "hdfs_folder_path": Param(  # <-- CORRECT CLASS
            default="/telecom/vertices/Customer",
            type="string",  # <-- 'type' is now valid
            title="HDFS Folder Path",
            description="The HDFS directory containing the data file to import.",
        )
    }, 
    doc_md="""
    ### Conditional Data Import to NebulaGraph from an HDFS Folder

    This DAG orchestrates data loading into NebulaGraph by automatically detecting the data file within a specified
    HDFS folder and choosing the import tool based on its extension.

    **Workflow:**
    1. The DAG is triggered with a parameter for the HDFS **folder path**.
    2. A branch operator executes an `hdfs dfs -ls` command to find files in that folder.
    3. It filters out temporary/system files (like `_SUCCESS`) and selects the first valid data file.
    4. **If the file is `.csv`:** It runs the `nebula-importer` using the `DockerOperator`.
    5. **Otherwise:** It submits a Spark job using `nebula-exchange` via the `SparkSubmitOperator`.
    """,
) as dag:
    find_and_branch_task = BranchPythonOperator(
        task_id="find_file_and_branch",
        python_callable=find_file_and_get_branch_task,
    )

    importer_task = DockerOperator(
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

    exchange_task = SparkSubmitOperator(
        task_id="submit_nebula_exchange_job",
        conn_id="spark-spark",
        application="/opt/spark_apps/nebula-exchange_spark_2.4-3.8.0.jar",
        java_class="com.vesoft.nebula.exchange.Exchange",
        application_args=["--config", "/opt/spark_apps/telecom_nebula_exchange.conf"],
        verbose=True,
    )

    find_and_branch_task >> [importer_task, exchange_task]
