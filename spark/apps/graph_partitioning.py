import os
import sys
import json
import logging
import tempfile
import traceback 
from collections import defaultdict
from typing import Dict, Any

import pandas as pd
import torch
import numpy as np
from hdfs import InsecureClient 

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, LongType, IntegerType, FloatType, BooleanType
from pyspark.sql.functions import pandas_udf, PandasUDFType 
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

def add_temporal_train_test_split(df: DataFrame, train_perc: float = 0.7, val_perc: float = 0.15) -> DataFrame:
    """
    Adds train/validation/test split masks based on a temporal column.
    This version correctly handles TimestampType by converting to Unix timestamps
    for quantile calculation, making it compatible with Spark 2.x.

    Args:
        df (DataFrame): DataFrame containing a 'timestamp' column.
        train_perc (float): The percentage of data for the training set.
        val_perc (float): The percentage of data for the validation set.

    Returns:
        DataFrame: The original DataFrame with added boolean mask columns:
                   'train_mask', 'val_mask', 'test_mask'.
    """
    logger.info(f"Creating temporal splits: {train_perc*100}% train, {val_perc*100}% val...")
    
    df_with_unix_ts = df.withColumn("unix_ts", F.col("timestamp").cast("long"))

    # Determine the numeric split points from the Unix timestamp column
    split_points = df_with_unix_ts.approxQuantile("unix_ts", [train_perc, train_perc + val_perc], 0.01)
    train_end_ts = split_points[0]
    val_end_ts = split_points[1]

    logger.info(f"Train data ends at Unix timestamp: {train_end_ts}")
    logger.info(f"Validation data ends at Unix timestamp: {val_end_ts}")

    df_with_masks = df_with_unix_ts.withColumn(
        "train_mask", F.col("unix_ts") <= train_end_ts
    ).withColumn(
        "val_mask", (F.col("unix_ts") > train_end_ts) & (F.col("unix_ts") <= val_end_ts)
    ).withColumn(
        "test_mask", F.col("unix_ts") > val_end_ts
    ).drop("unix_ts") 
    
    # Show distribution
    logger.info("Split distribution:")
    df_with_masks.groupBy("train_mask", "val_mask", "test_mask").count().show()
    
    return df_with_masks

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

def save_partitions_for_pyg(spark: SparkSession, enriched_nodes_df: DataFrame, edges_df: DataFrame,
                           total_num_nodes: int, total_num_edges: int, config: Dict[str, Any]):
    """
    Formats and saves partitioned graph data, including features, labels, and masks.
    """
    logger.info("Starting PyG-compatible save of partitions to HDFS.")
    output_path = config["output_path"]

    # --- 1. Create and Save Global Mapping Files (Unchanged) ---
    logger.info("Generating and saving global node and edge maps.")
    threshold = config.get("driver_memory_threshold", 100_000_000) # Use .get for safety
    if total_num_nodes > threshold:
        logger.warning(f"WARNING: Number of nodes ({total_num_nodes}) exceeds threshold ({threshold}).")
        logger.warning("Collecting node map to driver may cause OutOfMemoryError. Increase --driver-memory.")
    
    node_map_pd = enriched_nodes_df.select("global_node_id", "partition_id").toPandas()
    node_map_tensor = torch.full((total_num_nodes,), -1, dtype=torch.long)
    node_map_tensor[node_map_pd['global_node_id'].values] = torch.tensor(node_map_pd['partition_id'].values, dtype=torch.long)

    if total_num_edges > threshold:
        logger.warning(f"WARNING: Number of edges ({total_num_edges}) exceeds threshold ({threshold}).")
        logger.warning("Collecting edge map to driver may cause OutOfMemoryError. Increase --driver-memory.")
    
    node_partition_map = enriched_nodes_df.select("id", "partition_id")
    edges_with_partition = edges_df.join(node_partition_map, edges_df.src == node_partition_map.id, "inner")
    edge_map_pd = edges_with_partition.select("global_edge_id", "partition_id").toPandas()
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

    # --- 2. Prepare Data for Distributed Processing (Add mask columns) ---
    nodes_for_grouping = enriched_nodes_df.select(
        "partition_id", F.lit("node").alias("type"), "global_node_id", 
        "features", "label", "train_mask", "val_mask", "test_mask", # <-- ADDED MASKS and LABEL
        F.lit(None).cast(LongType()).alias("src_global"), F.lit(None).cast(LongType()).alias("dst_global"), F.lit(None).cast(LongType()).alias("global_edge_id")
    )
    # The rest of data prep is unchanged
    intra_partition_edges = (
        edges_df.join(enriched_nodes_df.alias("src_map"), F.col("src_global") == F.col("src_map.global_node_id"))
        .join(enriched_nodes_df.alias("dst_map"), F.col("dst_global") == F.col("dst_map.global_node_id"))
        .where(F.col("src_map.partition_id") == F.col("dst_map.partition_id"))
        .select(F.col("src_map.partition_id").alias("partition_id"), "src_global", "dst_global", "global_edge_id")
    )
    edges_for_grouping = intra_partition_edges.select(
        "partition_id", F.lit("edge").alias("type"), F.lit(None).alias("global_node_id"),
        F.lit(None).cast(ArrayType(FloatType())).alias("features"), F.lit(None).cast(IntegerType()).alias("label"),
        F.lit(None).cast(BooleanType()).alias("train_mask"), F.lit(None).cast(BooleanType()).alias("val_mask"), F.lit(None).cast(BooleanType()).alias("test_mask"),
        "src_global", "dst_global", "global_edge_id"
    )
    grouped_data = nodes_for_grouping.unionByName(edges_for_grouping)

    # --- 3. Define and Execute Pandas UDF (Updated to save labels and masks) ---
    b_total_num_nodes = spark.sparkContext.broadcast(total_num_nodes)
    namenode_host = config["hdfs_base_uri"].split('//')[1].split(':')[0]
    b_webhdfs_url = spark.sparkContext.broadcast(f"http://{namenode_host}:50070")
    b_output_path = spark.sparkContext.broadcast(output_path)
    b_hdfs_user = spark.sparkContext.broadcast(config.get("hdfs_user", "root"))
    
    result_schema = StructType([
        StructField("partition_id", IntegerType()), StructField("success", BooleanType())
    ])

    def process_and_save_partition_fn(pdf: pd.DataFrame) -> pd.DataFrame:
        if pdf.empty: return pd.DataFrame({'partition_id': [], 'success': []})
        partition_id = int(pdf['partition_id'].iloc[0])
        try:
            client = InsecureClient(b_webhdfs_url.value, user=b_hdfs_user.value)
            with tempfile.TemporaryDirectory() as tmpdir:
                nodes_pd = pdf[pdf['type'] == 'node'].sort_values('global_node_id').reset_index()
                
                feature_array = np.array(nodes_pd['features'].tolist(), dtype=np.float32)
                label_array = nodes_pd['label'].to_numpy(dtype=np.int64)
                train_mask_array = nodes_pd['train_mask'].to_numpy(dtype=np.bool_)
                val_mask_array = nodes_pd['val_mask'].to_numpy(dtype=np.bool_)
                test_mask_array = nodes_pd['test_mask'].to_numpy(dtype=np.bool_)
                
                feats_dict = {
                    'x': torch.from_numpy(feature_array),
                    'y': torch.from_numpy(label_array).to(torch.long),
                    'train_mask': torch.from_numpy(train_mask_array),
                    'val_mask': torch.from_numpy(val_mask_array),
                    'test_mask': torch.from_numpy(test_mask_array)
                }

                node_feats = {
                    'global_id': torch.tensor(nodes_pd['global_node_id'].values, dtype=torch.long),
                    'feats': feats_dict
                }

                node_feats_path = os.path.join(tmpdir, "node_feats.pt")
                torch.save(node_feats, node_feats_path)
                
                edges_pd = pdf[pdf['type'] == 'edge'].sort_values('dst_global').reset_index()
                if not edges_pd.empty:
                    graph = {'row': torch.tensor(edges_pd['src_global'].values, dtype=torch.long), 'col': torch.tensor(edges_pd['dst_global'].values, dtype=torch.long), 'edge_id': torch.tensor(edges_pd['global_edge_id'].values, dtype=torch.long), 'size': (b_total_num_nodes.value, b_total_num_nodes.value)}
                else:
                    graph = {'row': torch.empty(0, dtype=torch.long), 'col': torch.empty(0, dtype=torch.long), 'edge_id': torch.empty(0, dtype=torch.long), 'size': (b_total_num_nodes.value, b_total_num_nodes.value)}
                graph_path = os.path.join(tmpdir, "graph.pt")
                torch.save(graph, graph_path)
                
                edge_feats_path = os.path.join(tmpdir, "edge_feats.pt")
                torch.save(defaultdict(), edge_feats_path)

                hdfs_partition_dir = f"{b_output_path.value}/part_{partition_id}"
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

def run_test_2():
    """
    Main function modified to be a self-contained test for the 
    save_partitions_for_pyg function.
    
    It creates mock data in memory and calls the function to be tested,
    bypassing all external data sources.
    """
    # Temporarily modify config for local testing
    test_config = CONFIG.copy()
    test_config["spark_master_url"] = "local[*]"
    test_config["output_path"] = "/tmp/pyg_dataset_TEST"
    test_config["partitions"] = 2

    spark = None
    try:
        spark = create_spark_session(test_config)
        
        logger.info("--- Starting Test for save_partitions_for_pyg ---")

        # --- 1. Create Mock Enriched Nodes DataFrame ---
        logger.info("Step 1: Creating mock enriched_nodes_df...")
        
        # Ensure all feature vectors have the SAME length (e.g., 8)
        feature_dim = 8
        mock_node_data = [
            # id, global_node_id, partition_id, features, label, train, val, test
            ('CUST-00000001', 0, 0, [float(v) for v in np.random.rand(feature_dim)], 1, True, False, False),
            ('CUST-00000002', 1, 0, [float(v) for v in np.random.rand(feature_dim)], 0, True, False, False),
            ('CUST-00000003', 2, 0, [float(v) for v in np.random.rand(feature_dim)], 0, True, False, False),
            ('CUST-00000004', 3, 0, [float(v) for v in np.random.rand(feature_dim)], 1, True, False, False),
            ('CUST-00000005', 4, 0, [float(v) for v in np.random.rand(feature_dim)], 0, True, False, False),
            ('CUST-00000006', 5, 1, [float(v) for v in np.random.rand(feature_dim)], 0, True, False, False),
            ('CUST-00000007', 6, 1, [float(v) for v in np.random.rand(feature_dim)], 1, False, True, False),
            ('CUST-00000008', 7, 1, [float(v) for v in np.random.rand(feature_dim)], 0, False, True, False),
            ('CUST-00000009', 8, 1, [float(v) for v in np.random.rand(feature_dim)], 1, False, False, True),
            ('CUST-00000010', 9, 1, [float(v) for v in np.random.rand(feature_dim)], 0, False, False, True),
        ]
        
        node_schema = StructType([
            StructField("id", StringType(), False),
            StructField("global_node_id", LongType(), False),
            StructField("partition_id", IntegerType(), False),
            StructField("features", ArrayType(FloatType()), False),
            StructField("label", IntegerType(), False),
            StructField("train_mask", BooleanType(), False),
            StructField("val_mask", BooleanType(), False),
            StructField("test_mask", BooleanType(), False),
        ])

        enriched_nodes_df = spark.createDataFrame(mock_node_data, schema=node_schema)
        
        logger.info("Mock enriched_nodes_df created successfully:")
        enriched_nodes_df.show()

        # --- 2. Create Mock Edges DataFrame ---
        logger.info("Step 2: Creating mock edges_df...")
        mock_edge_data = [
            # src, dst, src_global, dst_global, global_edge_id
            ('CUST-00000001', 'CUST-00000002', 0, 1, 0),
            ('CUST-00000002', 'CUST-00000003', 1, 2, 1),
            ('CUST-00000004', 'CUST-00000005', 3, 4, 2), # Intra-partition edge (0)
            ('CUST-00000005', 'CUST-00000006', 4, 5, 3), # Inter-partition edge
            ('CUST-00000007', 'CUST-00000008', 6, 7, 4), # Intra-partition edge (1)
            ('CUST-00000009', 'CUST-00000010', 8, 9, 5), # Intra-partition edge (1)
            ('CUST-00000001', 'CUST-00000008', 0, 7, 6), # Inter-partition edge
        ]
        
        edge_schema = StructType([
            StructField("src", StringType(), False),
            StructField("dst", StringType(), False),
            StructField("src_global", LongType(), False),
            StructField("dst_global", LongType(), False),
            StructField("global_edge_id", LongType(), False),
        ])

        edges_df = spark.createDataFrame(mock_edge_data, schema=edge_schema)
        
        logger.info("Mock edges_df created successfully:")
        edges_df.show()

        # --- 3. Calculate Inputs and Call the Function ---
        logger.info("Step 3: Calculating inputs and calling save_partitions_for_pyg...")
        total_num_nodes = enriched_nodes_df.count()
        total_num_edges = edges_df.count()
        
        enriched_nodes_df.cache()
        edges_df.cache()

        save_partitions_for_pyg(
            spark=spark,
            enriched_nodes_df=enriched_nodes_df,
            edges_df=edges_df,
            total_num_nodes=total_num_nodes,
            total_num_edges=total_num_edges,
            config=test_config
        )

        logger.info("--- Test for save_partitions_for_pyg finished successfully. ---")
        logger.info(f"Check the output in HDFS at: {test_config['hdfs_base_uri']}{test_config['output_path']}")

    except Exception as e:
        logger.error("!!! An error occurred during the test !!!", exc_info=True)
        sys.exit(1)
    finally:
        if spark:
            logger.info("Stopping Spark session.")
            spark.stop()

def main():
    """Main execution function."""
    spark = None
    try:
        spark = create_spark_session(CONFIG)
        
        nodes_df, raw_edges_df = load_from_nebula(spark, CONFIG)
        nodes_df.cache()
        
        logger.info("Generating global 0-indexed IDs for nodes and edges...")
        node_window = Window.orderBy("id")
        node_id_map = nodes_df.select("id").distinct().withColumn("global_node_id", F.row_number().over(node_window) - 1)
        nodes_with_global_id = nodes_df.join(node_id_map, "id", "inner")
        
        edges_with_global_nodes = raw_edges_df.join(node_id_map.alias("src_map"), F.col("src") == F.col("src_map.id")) \
                                              .join(node_id_map.alias("dst_map"), F.col("dst") == F.col("dst_map.id")) \
                                              .select("src_map.global_node_id as src_global", "dst_map.global_node_id as dst_global", "src", "dst")
        
        edge_window = Window.orderBy("src_global", "dst_global")
        edges_df = edges_with_global_nodes.withColumn("global_edge_id", F.row_number().over(edge_window) - 1)
        edges_df.cache()
        
        total_num_nodes = nodes_with_global_id.count()
        total_num_edges = edges_df.count()
        logger.info(f"Total nodes: {total_num_nodes}, Total edges: {total_num_edges}")
        
        partitioned_nodes_df = partition_graph(nodes_df, raw_edges_df, CONFIG["partitions"])
        nodes_df.unpersist()

        partitioned_nodes_with_global_id = partitioned_nodes_df.join(nodes_with_global_id, "id", "inner")
        
        enriched_nodes_df = enrich_data_from_cassandra(spark, partitioned_nodes_with_global_id, CONFIG)
        
        final_nodes_df = add_temporal_train_test_split(enriched_nodes_df)
        final_nodes_df.cache()

        save_partitions_for_pyg(spark, final_nodes_df, edges_df, total_num_nodes, total_num_edges, CONFIG)
        
        edges_df.unpersist()
        final_nodes_df.unpersist()

        logger.info("GNN data preparation pipeline finished successfully.")

    except Exception as e:
        logger.error(f"An error occurred during the Spark job: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if spark:
            logger.info("Stopping Spark session.")
            spark.stop()
            
if __name__ == "__main__":
    run_test_2()
