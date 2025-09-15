import os
import sys
import json
import logging
import tempfile
import subprocess
from collections import defaultdict
from typing import Dict, Any

import pandas as pd
import torch
import numpy as np
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, LongType, IntegerType, FloatType, BooleanType
from pyspark.sql.functions import pandas_udf, PandasUDFType
from hdfs import InsecureClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration Section (Unchanged) ---
CONFIG = {
    "app_name": "GNN_Data_Prep_Pipeline_TEST",
    "partitions": 2,
    "spark_master_url": "spark://spark-master:7077", # Use local mode for easier testing
    "spark_packages": [
        "com.datastax.spark:spark-cassandra-connector_2.11:2.4.3",
        "graphframes:graphframes:0.8.1-spark2.4-s_2.11",
    ],
    "hdfs_base_uri": "hdfs://namenode:8020",
    "output_path": "/tmp/pyg_dataset_TEST",
    "checkpoint_dir": "/tmp/spark_checkpoints_TEST",
    "cassandra": {
        "host": "cassandra",
        "port": "9042",
        "keyspace": "feature_store",
        "table": "customer_features"
    }
}

# --- Utility Functions (Unchanged) ---
def create_spark_session(config: Dict[str, Any]) -> SparkSession:
    """Initializes and returns a SparkSession."""
    logger.info("Initializing Spark session...")
    spark_builder = (
        SparkSession.builder.appName(config["app_name"])
        .master(config["spark_master_url"])
        .config("spark.jars.packages", ",".join(config["spark_packages"]))
        .config(f"spark.cassandra.connection.host", config["cassandra"]["host"])
        .config(f"spark.cassandra.connection.port", config["cassandra"]["port"])
    )
    spark = spark_builder.getOrCreate()
    spark.sparkContext.setCheckpointDir(config["checkpoint_dir"])
    logger.info("Spark session created successfully.")
    return spark

def enrich_data_from_cassandra(spark: SparkSession, partitioned_nodes_df: DataFrame, config: Dict[str, Any]) -> DataFrame:
    """Enriches the partitioned node data by joining with features and labels from Cassandra."""
    logger.info(f"Enriching node data from Cassandra table: {config['cassandra']['keyspace']}.{config['cassandra']['table']}")
    # NOTE: For a unit test, this will try to connect to a real Cassandra.
    # Ensure it's running or mock this function if needed.
    features_df = (
        spark.read.format("org.apache.spark.sql.cassandra")
        .options(table=config['cassandra']['table'], keyspace=config['cassandra']['keyspace'])
        .load()
        .withColumnRenamed("cust_id", "id")
    )
    enriched_nodes_df = partitioned_nodes_df.join(features_df, "id", "inner")
    
    count = enriched_nodes_df.count()
    logger.info(f"Successfully enriched {count} nodes from Cassandra.")
    if count == 0:
        logger.warning("WARNING: No matching nodes found in Cassandra. Ensure your sample data is inserted correctly.")
    
    return enriched_nodes_df

def _write_to_hdfs_from_driver(local_path: str, hdfs_path: str, config: Dict[str, Any]):
    """
    Uploads a local file to HDFS using the native 'hdfs' Python library.
    This avoids dependency on the 'hadoop' shell command.
    """
    namenode_host = config["hdfs_base_uri"].split('//')[1].split(':')[0]
    webhdfs_url = f"http://{namenode_host}:50070"
    
    # The 'user' parameter can be set to the user running the script, e.g., 'root'
    client = InsecureClient(webhdfs_url, user='root')

    logger.info(f"Uploading file '{local_path}' to HDFS at '{hdfs_path}' via WebHDFS.")
    
    try:
        # The client automatically handles creating parent directories.
        client.upload(
            hdfs_path=hdfs_path,
            local_path=local_path,
            overwrite=True
        )
        logger.info("Upload successful.")
    except Exception as e:
        logger.error(f"Failed to upload to HDFS via WebHDFS. Error: {e}")
        raise e

def save_partitions_for_pyg(spark: SparkSession, enriched_nodes_df: DataFrame, edges_df: DataFrame, total_num_nodes: int, config: Dict[str, Any]):
    """
    Formats and saves partitioned graph data to HDFS in a PyG-compatible format.
    This version is corrected to use the 'hdfs' library instead of subprocess.
    """
    logger.info("Starting PyG-compatible save of partitions to HDFS.")
    output_path = config["output_path"]
    hdfs_base_uri = config["hdfs_base_uri"]

    # --- 1. Create and Save Global Mapping Files (from Driver) ---
    logger.info("Generating and saving global node and edge maps.")
    node_map_pd = enriched_nodes_df.select("global_node_id", "partition_id").toPandas()
    node_map_tensor = torch.full((total_num_nodes,), -1, dtype=torch.long)
    node_map_tensor[node_map_pd['global_node_id'].values] = torch.tensor(node_map_pd['partition_id'].values, dtype=torch.long)

    node_partition_map = enriched_nodes_df.select("id", "partition_id")
    edges_with_partition = edges_df.join(node_partition_map, edges_df.src == node_partition_map.id, "inner")
    edge_map_pd = edges_with_partition.select("global_edge_id", "partition_id").toPandas()

    total_num_edges = edges_df.count()
    edge_map_tensor = torch.full((total_num_edges,), -1, dtype=torch.long)
    edge_map_tensor[edge_map_pd['global_edge_id'].values] = torch.tensor(edge_map_pd['partition_id'].values, dtype=torch.long)

    with tempfile.TemporaryDirectory() as tmpdir:
        torch.save(node_map_tensor, os.path.join(tmpdir, "node_map.pt"))
        _write_to_hdfs_from_driver(os.path.join(tmpdir, "node_map.pt"), f"{output_path}/node_map.pt", config)

        torch.save(edge_map_tensor, os.path.join(tmpdir, "edge_map.pt"))
        _write_to_hdfs_from_driver(os.path.join(tmpdir, "edge_map.pt"), f"{output_path}/edge_map.pt", config)

        meta = {'num_parts': config["partitions"], 'is_hetero': False, 'node_types': None, 'edge_types': None, 'is_sorted': True}
        with open(os.path.join(tmpdir, "META.json"), 'w') as f:
            json.dump(meta, f)
        _write_to_hdfs_from_driver(os.path.join(tmpdir, "META.json"), f"{output_path}/META.json", config)

    # --- 2. Prepare Data for Distributed Processing (Unchanged) ---
    node_map = enriched_nodes_df.select("id", "global_node_id", "partition_id")
    intra_partition_edges = (
        edges_df.join(node_map.alias("src_map"), F.col("src_global") == F.col("src_map.global_node_id"))
        .join(node_map.alias("dst_map"), F.col("dst_global") == F.col("dst_map.global_node_id"))
        .where(F.col("src_map.partition_id") == F.col("dst_map.partition_id"))
        .select(F.col("src_map.partition_id").alias("partition_id"), "src_global", "dst_global", "global_edge_id")
    )

    nodes_for_grouping = enriched_nodes_df.select(
        "partition_id", F.lit("node").alias("type"), "global_node_id", "features", "label",
        F.lit(None).cast(LongType()).alias("src_global"), F.lit(None).cast(LongType()).alias("dst_global"), F.lit(None).cast(LongType()).alias("global_edge_id")
    )
    edges_for_grouping = intra_partition_edges.select(
        "partition_id", F.lit("edge").alias("type"), F.lit(None).alias("global_node_id"),
        F.lit(None).cast(ArrayType(FloatType())).alias("features"), F.lit(None).cast(IntegerType()).alias("label"),
        "src_global", "dst_global", "global_edge_id"
    )
    grouped_data = nodes_for_grouping.unionByName(edges_for_grouping)

    # --- 3. Define and Execute Pandas UDF with the HDFS library ---
    b_total_num_nodes = spark.sparkContext.broadcast(total_num_nodes)

    # Broadcast HDFS connection info to all executors
    namenode_host = config["hdfs_base_uri"].split('//')[1].split(':')[0]
    b_webhdfs_url = spark.sparkContext.broadcast(f"http://{namenode_host}:50070")
    b_output_path = spark.sparkContext.broadcast(output_path)

    result_schema = StructType([
        StructField("partition_id", IntegerType()), StructField("success", BooleanType())
    ])

    def process_and_save_partition_fn(pdf: pd.DataFrame) -> pd.DataFrame:
        if pdf.empty:
            return pd.DataFrame({'partition_id': [], 'success': []})

        partition_id = int(pdf['partition_id'].iloc[0])
        total_nodes = b_total_num_nodes.value
        webhdfs_url = b_webhdfs_url.value
        output_path = b_output_path.value

        try:
            # Each executor creates its own HDFS client
            client = InsecureClient(webhdfs_url, user='root')

            with tempfile.TemporaryDirectory() as tmpdir:
                # Process nodes and edges as before...
                nodes_pd = pdf[pdf['type'] == 'node'].sort_values('global_node_id').reset_index()
                node_feats = {'global_id': torch.tensor(nodes_pd['global_node_id'].values, dtype=torch.long), 'feats': {'x': torch.tensor(np.stack(nodes_pd['features'].values), dtype=torch.float)}}
                node_feats_path = os.path.join(tmpdir, "node_feats.pt")
                torch.save(node_feats, node_feats_path)

                edges_pd = pdf[pdf['type'] == 'edge'].sort_values('dst_global').reset_index()
                if not edges_pd.empty:
                    graph = {'row': torch.tensor(edges_pd['src_global'].values, dtype=torch.long), 'col': torch.tensor(edges_pd['dst_global'].values, dtype=torch.long), 'edge_id': torch.tensor(edges_pd['global_edge_id'].values, dtype=torch.long), 'size': (total_nodes, total_nodes)}
                else:
                    graph = {'row': torch.empty(0, dtype=torch.long), 'col': torch.empty(0, dtype=torch.long), 'edge_id': torch.empty(0, dtype=torch.long), 'size': (total_nodes, total_nodes)}
                graph_path = os.path.join(tmpdir, "graph.pt")
                torch.save(graph, graph_path)

                edge_feats_path = os.path.join(tmpdir, "edge_feats.pt")
                torch.save(defaultdict(), edge_feats_path)

                # --- Use HDFS client for uploading ---
                hdfs_partition_dir = f"{output_path}/part_{partition_id}"
                client.upload(f"{hdfs_partition_dir}/node_feats.pt", node_feats_path, overwrite=True)
                client.upload(f"{hdfs_partition_dir}/graph.pt", graph_path, overwrite=True)
                client.upload(f"{hdfs_partition_dir}/edge_feats.pt", edge_feats_path, overwrite=True)

            success = True
        except Exception as e:
            print(f"ERROR processing partition {partition_id}: {e}")
            success = False
        return pd.DataFrame([{'partition_id': partition_id, 'success': success}])

    process_and_save_partition_udf = pandas_udf(process_and_save_partition_fn, result_schema, PandasUDFType.GROUPED_MAP)

    logger.info("Executing save operation on all partitions...")
    result = grouped_data.groupBy("partition_id").apply(process_and_save_partition_udf)

    failed_partitions = result.where(F.col("success") == False).count()
    if failed_partitions > 0:
        logger.error(f"{failed_partitions} partitions failed to save. Check executor logs for details.")
    else:
        logger.info("All partitions processed and saved successfully.")

def run_test():
    """
    Runs a targeted test for the enrichment and save functions using a small,
    in-memory sample of data. This version correctly prepares the data schema.
    """
    spark = create_spark_session(CONFIG)

    logger.info("Step 1: Creating mock data for testing...")
    sample_ids = [(f'CUST-{i:08}',) for i in range(1, 11)]
    mock_raw_nodes_df = spark.createDataFrame(sample_ids, ["id"])

    mock_raw_edges_data = [
        ('CUST-00000002', 'CUST-00000004'), ('CUST-00000004', 'CUST-00000005'),
        ('CUST-00000006', 'CUST-00000007'), ('CUST-00000007', 'CUST-00000008'),
        ('CUST-00000009', 'CUST-00000010'), ('CUST-00000005', 'CUST-00000006')
    ]
    mock_raw_edges_df = spark.createDataFrame(mock_raw_edges_data, ["src", "dst"])
    
    logger.info("Step 1a: Creating global, 0-indexed IDs...")
    
    # Define a window to order the nodes deterministically
    node_window = Window.orderBy("id")
    
    # Use row_number() which starts at 1, so subtract 1 to make it 0-indexed
    node_id_map = mock_raw_nodes_df.withColumn("global_node_id", F.row_number().over(node_window) - 1)
    
    # No need for a separate join, the map is the DataFrame with the new ID
    nodes_with_global_id = node_id_map

    # Join to get global IDs for edges
    edges_with_global_nodes = mock_raw_edges_df.join(node_id_map.alias("src_map"), F.col("src") == F.col("src_map.id")) \
                                           .join(node_id_map.alias("dst_map"), F.col("dst") == F.col("dst_map.id")) \
                                           .select(
                                               F.col("src_map.global_node_id").alias("src_global"),
                                               F.col("dst_map.global_node_id").alias("dst_global"),
                                               "src", "dst"
                                           )

    # Define a window to create consecutive edge IDs
    edge_window = Window.orderBy("src_global", "dst_global")
    edges_df = edges_with_global_nodes.withColumn("global_edge_id", F.row_number().over(edge_window) - 1)

    # Create mock partitions
    mock_partitioned_nodes_df = nodes_with_global_id.withColumn(
        "partition_id", F.when(F.col("id") < "CUST-00000006", 0).otherwise(1)
    )

    logger.info("Mock partitioned data with correct global IDs:")
    mock_partitioned_nodes_df.show()
    
    logger.info("\nStep 2: Testing enrich_data_from_cassandra...")
    enriched_nodes_df = enrich_data_from_cassandra(spark, mock_partitioned_nodes_df, CONFIG)
    enriched_nodes_df.cache()
    
    logger.info("Enriched data schema:")
    enriched_nodes_df.printSchema()
    enriched_nodes_df.show()
    
    total_nodes = enriched_nodes_df.count()

    logger.info("\nStep 3: Testing save_partitions_for_pyg...")
    save_partitions_for_pyg(spark, enriched_nodes_df, edges_df, total_nodes, CONFIG)

    logger.info("Test finished successfully.")
    enriched_nodes_df.unpersist()
    spark.stop()

if __name__ == "__main__":
    run_test()
