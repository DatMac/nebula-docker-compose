from __future__ import annotations

import pendulum
from docker.types import Mount

from airflow.models.dag import DAG
from airflow.providers.docker.operators.docker import DockerOperator

HOST_IMPORTER_CONFIG_PATH = "/home/macquangdat2412/nebula-docker-compose/nebula_importer/config/hdfs_config.yaml"
CONTAINER_CONFIG_PATH = "/configs/hdfs_config.yaml"
HOST_LOG_PATH = "/home/macquangdat2412/nebula-docker-compose/nebula_importer/logs"

# --- DAG DEFINITION ---
with DAG(
    dag_id="import_hdfs_to_nebula_with_docker",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    schedule=None,  # Run manually, or set to "@daily", "0 0 * * *", etc.
    catchup=False,
    tags=["nebula", "docker", "importer", "hdfs"],
    doc_md="""
    ### Import HDFS Data to NebulaGraph using Nebula Importer

    This DAG uses the DockerOperator to run the `vesoft/nebula-importer` container.

    **Workflow:**
    1. The operator starts the `nebula-importer:v3.4.0` container.
    2. It connects the container to the `nebula-net` network, allowing it to
       communicate with both the Nebula `graphd` service and the HDFS `namenode`.
    3. It mounts the local YAML configuration file into the container.
    4. The importer reads the config, connects to HDFS, processes the data, and
       loads it into NebulaGraph.

    **Prerequisites:**
    - You must update the `HOST_IMPORTER_CONFIG_PATH` variable in this DAG file.
    - The configuration file itself (`importer_from_hdfs.yaml`) must be correctly
      filled out with your HDFS path, Nebula space, and data schema.
    """,
) as dag:
    
    import_task = DockerOperator(
        task_id="run_nebula_importer_from_hdfs",
        image="vesoft/nebula-importer:v4",
        docker_conn_id=None,
        command=f"--config {CONTAINER_CONFIG_PATH}",
        network_mode="nebula-docker-compose_nebula-net",
        mounts=[
            Mount(
                source=HOST_IMPORTER_CONFIG_PATH,
                target=CONTAINER_CONFIG_PATH,
                type="bind",
                read_only=True
            ),
            Mount(
                source=HOST_LOG_PATH,
                target="/configs/logs",
                type="bind"
            )
        ],
        auto_remove=True,
        tty=True,
    )
