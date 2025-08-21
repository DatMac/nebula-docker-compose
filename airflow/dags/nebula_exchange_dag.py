from __future__ import annotations

import pendulum

from airflow.models.dag import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

# --- DAG Definition ---
with DAG(
    dag_id="test_spark_nebula_exchange",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    schedule=None,  # This DAG is not scheduled and runs only when manually triggered.
    catchup=False,  # Do not run for past, un-run schedules.
    tags=["spark", "nebula", "test"],
    doc_md="""
    ### Test Spark Nebula Exchange DAG

    This DAG submits a Spark job using the Nebula Exchange JAR to process data.
    It reads a configuration file and loads data into NebulaGraph.
    
    **Connection Info:**
    - Assumes an Airflow connection with `conn_id` of **spark_default**.
    - The connection URL should point to the Spark Master: `spark://spark-master:7077`.
      (This is already configured by the `AIRFLOW_CONN_SPARK_DEFAULT` environment
      variable in your `docker-compose.txt` file).

    **File Paths:**
    - The JAR file and config file are expected to be in `/opt/spark_apps/` inside the Airflow container.
    """,
) as dag:
    # --- Task Definition ---
    submit_spark_job = SparkSubmitOperator(
        task_id="submit_nebula_exchange_job",
        # Use the default Spark connection configured in your docker-compose file.
        conn_id="spark-spark",
        
        # Corresponds to the main application JAR file.
        # Path inside the Airflow container.
        application="/opt/spark_apps/nebula-exchange_spark_2.4-3.8.0.jar",
        
        # Corresponds to the `--class` argument.
        java_class="com.vesoft.nebula.exchange.Exchange",
        
        # Corresponds to arguments passed to the application.
        application_args=["--config", "/opt/spark_apps/parquet.conf"],
        
        verbose=True,
    )
