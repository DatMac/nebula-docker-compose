from __future__ import annotations

import pendulum

from airflow.models.dag import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

with DAG(
    dag_id="naive_materialize_dag",
    schedule=None,
    start_date=pendulum.datetime(2025, 9, 27, tz="UTC"),
    catchup=False,
    doc_md="""
    ## One-Time Spark Job: HDFS to Cassandra

    This DAG submits a PySpark application that reads customer feature data from HDFS
    and writes it to a Cassandra table. It is configured to be triggered manually.
    """,
    tags=["spark", "cassandra", "onetime"],
) as dag:
    submit_spark_job = SparkSubmitOperator(
        task_id="submit_hdfs_to_cassandra_job",
        conn_id="spark-spark", 
        application="/opt/spark_apps/naive_materialize.py",
        name="HdfsToCassandraLoader",
        conf={
            "spark.jars.packages": "com.datastax.spark:spark-cassandra-connector_2.11:2.4.3,com.twitter:jsr166e:1.1.0"
        },
        packages="com.datastax.spark:spark-cassandra-connector_2.11:2.4.3,com.twitter:jsr166e:1.1.0",
        verbose=True,
    )
