from __future__ import annotations

import pendulum

from airflow.models.dag import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

SPARK_PACKAGES = [
    "com.vesoft.nebula-spark-connector:nebula-spark-connector_2.2:3.4.0",
    "com.datastax.spark:spark-cassandra-connector_2.11:2.4.3",
    "graphframes:graphframes:0.8.1-spark2.4-s_2.11",
    "com.twitter:jsr166e:1.1.0"
]

with DAG(
    dag_id="gcn_data_preparation_dag",
    schedule="@daily",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    is_paused_upon_creation=True,
    doc_md="""
    ## Daily GNN Data Preparation Pipeline

    This DAG orchestrates a complex daily Spark job to prepare graph data for Graph Neural Network (GNN) training.

    ### Pipeline Steps:
    1.  **Load Data**: Fetches nodes and edges from a NebulaGraph database.
    2.  **Enrich Features**: Joins the node data with feature tables stored in Cassandra.
    3.  **Graph Partitioning**: Uses the Label Propagation Algorithm (LPA) via GraphFrames to partition the graph.
    4.  **Temporal Split**: Adds boolean masks (`train_mask`, `val_mask`, `test_mask`) for a time-based data split.
    5.  **Format and Save**: Saves the final, partitioned data to HDFS in a format compatible with PyTorch Geometric's distributed loader.

    ### Important Considerations:
    - **Spark Driver Memory**: This job collects node and edge maps to the Spark driver. The driver instance must have sufficient memory to handle the entire graph's mapping metadata.
    - **Connectivity**: The Airflow workers and the Spark cluster must have network access to NebulaGraph, Cassandra, and HDFS.
    """,
    tags=["spark", "gnn", "nebula", "cassandra", "daily"],
) as dag:
    submit_gnn_prep_job = SparkSubmitOperator(
        task_id="submit_gnn_data_prep_job",
        conn_id="spark-spark",
        application="/opt/spark_apps/data_partitioner.py",
        name="GNN_Data_Prep_Pipeline",
        packages=",".join(SPARK_PACKAGES),
        conf={
            "spark.cassandra.connection.host": "cassandra",
            "spark.cassandra.connection.port": "9042",
            "spark.sql.shuffle.partitions": "200",
        },
        verbose=True,
    )
