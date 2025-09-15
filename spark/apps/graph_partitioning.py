import os
import sys
import json
import logging
import tempfile
from collections import defaultdict
from typing import Dict, Any

import pandas as pd
import torch
import numpy as np
from hdfs import InsecureClient # <-- ADDED IMPORT

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, LongType, IntegerType, FloatType, BooleanType
from pyspark.sql.functions import pandas_udf, PandasUDFType # <-- ADDED IMPORTS for Spark 2.4
from graphframes import GraphFrame

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration Section ---
# All configurable parameters are centralized here for easy management.
CONFIG = {
    "app_name": "GNN_Data_Prep_Pipeline",
    "partitions": 8,  # Number of partitions for the graph (N)

    "spark_master_url": "spark://spark-master:7077",

    "spark_packages": [
        "com.vesoft.nebula-spark-connector:nebula-spark-connector_2.2:3.4.0",
        "com.datastax.spark:spark-cassandra-connector_2.11:2.4.3",
        "graphframes:graphframes:0.8.1-spark2.4-s_2.11",
        "com.twitter:jsr166e:1.1.0"
    ],

    # HDFS Configuration
    "hdfs_base_uri": "hdfs://namenode:8020",
    "output_path": "/tmp/pyg_dataset",
    "checkpoint_dir": "/tmp/spark_checkpoints",

    # NebulaGraph Connection Details
    "nebula": {
        "meta_address": "metad0:9559,metad1:9559,metad2:9559",
        "space": "telecom",
        "user": "root",
        "password": "nebula",
        "vertex_label": "Customer",
        "vertex_return_cols": "cust_id",
    },

    # Cassandra Connection Details
    "cassandra": {
        "host": "cassandra",
        "port": "9042",
        "keyspace": "feature_store",
        "table": "customer_features"
    }
}


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


def load_from_nebula(spark: SparkSession, config: Dict[str, Any]) -> (DataFrame, DataFrame):
    """Loads vertices and ALL edges from NebulaGraph into Spark DataFrames."""
    logger.info("Creating plan to load vertices...")
    nodes_df = (
        spark.read.format("com.vesoft.nebula.connector.NebulaDataSource")
        .option("type", "vertex")
        .option("spaceName", config['nebula']['space'])
        .option("label", config['nebula']['vertex_label'])
        .option("returnCols", config['nebula']['vertex_return_cols'])
        .option("metaAddress", config['nebula']['meta_address'])
        .option("user", config["nebula"]["user"])
        .option("passwd", config["nebula"]["password"])
        .option("partitionNumber", 3)
        .load()
        .withColumnRenamed("_vertexId", "id")
    )

    logger.info(f"Creating plan to load all edge types from Nebula space '{config['nebula']['space']}'.")
    edge_labels = ["CALL", "SEND_SMS", "TRANSACTION"]
    all_edges_df = None

    for label in edge_labels:
        current_edges_df = (
            spark.read.format("com.vesoft.nebula.connector.NebulaDataSource")
            .option("type", "edge")
            .option("spaceName", config['nebula']['space'])
            .option("label", label)
            .option("returnCols", "") 
            .option("metaAddress", config['nebula']['meta_address'])
            .option("user", config["nebula"]["user"])
            .option("passwd", config["nebula"]["password"])
            .option("partitionNumber", 3)
            .load()
            .select(
                F.col("_srcId").alias("src"),
                F.col("_dstId").alias("dst")
            )
        )
        if all_edges_df is None:
            all_edges_df = current_edges_df
        else:
            all_edges_df = all_edges_df.union(current_edges_df)

    logger.info("Data loading plan from Nebula created.")
    return nodes_df, all_edges_df


def partition_graph(nodes_df: DataFrame, edges_df: DataFrame, num_partitions: int) -> DataFrame:
    """Partitions the graph using GraphFrames' Label Propagation Algorithm (LPA)."""
    logger.info("Partitioning graph using Label Propagation Algorithm...")
    g = GraphFrame(nodes_df, edges_df)
    partition_assignments = g.labelPropagation(maxIter=5)
    
    final_partitions_df = partition_assignments.withColumn(
        "partition_id", (F.col("label") % num_partitions).cast("int")
    ).select("id", "partition_id")

    logger.info("Graph partitioning complete. Partition distribution:")
    final_partitions_df.groupBy("partition_id").count().orderBy("partition_id").show()
    return final_partitions_df


def enrich_data_from_cassandra(spark: SparkSession, partitioned_nodes_df: DataFrame, config: Dict[str, Any]) -> DataFrame:
    """Enriches the partitioned node data by joining with features and labels from Cassandra."""
    logger.info(f"Enriching node data from Cassandra table: {config['cassandra']['keyspace']}.{config['cassandra']['table']}")
    features_df = (
        spark.read.format("org.apache.spark.sql.cassandra")
        .options(table=config['cassandra']['table'], keyspace=config['cassandra']['keyspace'])
        .load()
        .withColumnRenamed("cust_id", "id")
    )
    enriched_nodes_df = partitioned_nodes_df.join(features_df, "id", "inner")
    
    logger.info("Successfully created plan to enrich nodes.")
    return enriched_nodes_df

def _write_to_hdfs_from_driver(local_path: str, hdfs_path: str, config: Dict[str, Any]):
    """
    Uploads a local file to HDFS using the native 'hdfs' Python library.
    """
    namenode_host = config["hdfs_base_uri"].split('//')[1].split(':')[0]
    webhdfs_url = f"http://{namenode_host}:50070"
    client = InsecureClient(webhdfs_url, user='root')

    logger.info(f"Uploading file '{local_path}' to HDFS at '{hdfs_path}' via WebHDFS.")
    
    try:
        client.upload(hdfs_path=hdfs_path, local_path=local_path, overwrite=True)
        logger.info("Upload successful.")
    except Exception as e:
        logger.error(f"Failed to upload to HDFS via WebHDFS. Error: {e}")
        raise e

def save_partitions_for_pyg(spark: SparkSession, enriched_nodes_df: DataFrame, edges_df: DataFrame, total_num_nodes: int, config: Dict[str, Any]):
    """
    Formats and saves partitioned graph data to HDFS in a PyG-compatible format.
    This version uses the native hdfs library and is compatible with Spark 2.4.4.
    """
    logger.info("Starting PyG-compatible save of partitions to HDFS.")
    output_path = config["output_path"]

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

    # --- 2. Prepare Data for Distributed Processing ---
    intra_partition_edges = (
        edges_df.join(enriched_nodes_df.alias("src_map"), F.col("src_global") == F.col("src_map.global_node_id"))
        .join(enriched_nodes_df.alias("dst_map"), F.col("dst_global") == F.col("dst_map.global_node_id"))
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
            client = InsecureClient(webhdfs_url, user='root')
            with tempfile.TemporaryDirectory() as tmpdir:
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

def main():
    """Main execution function."""
    spark = None
    try:
        spark = create_spark_session(CONFIG)
        
        nodes_df, raw_edges_df = load_from_nebula(spark, CONFIG)
        nodes_df.cache()
        
        # --- Create global, consecutive, 0-indexed IDs ---
        logger.info("Generating global 0-indexed IDs for nodes and edges...")
        node_window = Window.orderBy("id")
        node_id_map = nodes_df.select("id").distinct().withColumn(
            "global_node_id", F.row_number().over(node_window) - 1
        )
        nodes_with_global_id = nodes_df.join(node_id_map, "id", "inner")
        
        edges_with_global_nodes = raw_edges_df.join(node_id_map.alias("src_map"), F.col("src") == F.col("src_map.id")) \
                                              .join(node_id_map.alias("dst_map"), F.col("dst") == F.col("dst_map.id")) \
                                              .select(
                                                  F.col("src_map.global_node_id").alias("src_global"),
                                                  F.col("dst_map.global_node_id").alias("dst_global"),
                                                  "src", "dst"
                                              )
        
        edge_window = Window.orderBy("src_global", "dst_global")
        edges_df = edges_with_global_nodes.withColumn("global_edge_id", F.row_number().over(edge_window) - 1)
        edges_df.cache()
        
        total_num_nodes = nodes_with_global_id.count()
        logger.info(f"Total nodes: {total_num_nodes}, Total edges: {edges_df.count()}")
        
        # Partition the graph using original string IDs
        partitioned_nodes_df = partition_graph(nodes_df, raw_edges_df, CONFIG["partitions"])
        nodes_df.unpersist()

        # Join partitioning results with the nodes that have global IDs
        partitioned_nodes_with_global_id = partitioned_nodes_df.join(nodes_with_global_id, "id", "inner")
        
        enriched_nodes_df = enrich_data_from_cassandra(spark, partitioned_nodes_with_global_id, CONFIG)
        enriched_nodes_df.cache()

        # Call the rewritten save function
        save_partitions_for_pyg(spark, enriched_nodes_df, edges_df, total_num_nodes, CONFIG)
       
        edges_df.unpersist()
        enriched_nodes_df.unpersist()

        logger.info("GNN data preparation pipeline finished successfully.")

    except Exception as e:
        logger.error(f"An error occurred during the Spark job: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if spark:
            logger.info("Stopping Spark session.")
            spark.stop()


if __name__ == "__main__":
    main()
