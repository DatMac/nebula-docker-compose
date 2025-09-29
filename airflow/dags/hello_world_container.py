from __future__ import annotations

import pendulum

from airflow.models.dag import DAG
from airflow.providers.docker.operators.docker import DockerOperator

# --- DAG Definition ---
with DAG(
    dag_id="test_docker_operator_hello_world",
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    schedule=None,  # This DAG is not scheduled and runs only when manually triggered.
    catchup=False,  # Do not run for past, un-run schedules.
    tags=["docker", "test", "example"],
    doc_md="""
    ### Test Docker Operator DAG

    This is a simple test DAG to confirm that the Airflow environment can successfully
    run a Docker container using the DockerOperator.

    It pulls and runs the basic `hello-world` container and prints its output to the logs.
    If this DAG runs successfully, it confirms that:
    1. The `apache-airflow-providers-docker` package is correctly installed.
    2. The Airflow container has access to the host's Docker socket.
    3. The `docker_default` connection is working.
    """,
) as dag:
    # --- Task Definition ---
    test_docker_task = DockerOperator(
        task_id="run_hello_world_container",
        docker_conn_id=None,
        image="hello-world",
        auto_remove=True,
        tty=True,
    )
