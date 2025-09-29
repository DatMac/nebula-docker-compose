from __future__ import annotations

import pendulum

from airflow.models.dag import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

with DAG(
    dag_id="incremental_materialize_dag",
    schedule="@daily",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    doc_md="""
    ## Daily Incremental Spark Job: HDFS to Cassandra

    This DAG orchestrates a daily Spark job that incrementally loads data from HDFS to Cassandra.

    - **Schedule**: Runs every day at midnight UTC.
    - **Logic**: For each run, it processes data from the previous day. For example, the run on September 28th will process data where the timestamp is between Sept 27th 00:00:00 and Sept 28th 00:00:00.
    - **Parameters**:
        - `start_timestamp`: Automatically set to the beginning of the data interval (e.g., `2025-09-27 00:00:00`).
        - `end_timestamp`: Automatically set to the end of the data interval (e.g., `2025-09-28 00:00:00`).
    """,
    tags=["spark", "cassandra", "daily", "incremental"],
) as dag:
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
    )
