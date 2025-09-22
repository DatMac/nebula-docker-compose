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
    "partitions": 100,  # Number of partitions for the graph (N)

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
        .config(f"spark.local.dir", "/opt/bitnami/spark/tmp-dir")
        .config(f"spark.sql.shuffle.partitions", 100)
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "1000")
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
    partition_assignments = g.labelPropagation(maxIter=1)
    
    final_partitions_df = partition_assignments.withColumn(
        "partition_id", (F.col("label") % num_partitions).cast("int")
    ).select("id", "partition_id")

    logger.info("Graph partitioning complete. Partition distribution:")
    final_partitions_df.groupBy("partition_id").count().orderBy("partition_id").show(200)
    return final_partitions_df


def enrich_data_from_cassandra(spark: SparkSession, partitioned_nodes_df: DataFrame, config: Dict[str, Any]) -> DataFrame:
    """Enriches the partitioned node data by joining with features and labels from Cassandra."""
    logger.info(f"Enriching node data from Cassandra table: {config['cassandra']['keyspace']}.{config['cassandra']['table']}")
    features_df = (
        spark.read.format("org.apache.spark.sql.cassandra")
        .options(table=config['cassandra']['table'], keyspace=config['cassandra']['keyspace'])
        .load()
    )
    enriched_nodes_df = partitioned_nodes_df.join(features_df, "cust_id", "inner")
    if enriched_nodes_df.rdd.isEmpty():
        raise ValueError(
            "The INNER join on 'cust_id' between nodes from Nebula and features from Cassandra "
            "resulted in an empty DataFrame. Please check for data consistency in the 'cust_id' "
            "column across both data sources."
        )
    
    logger.info("Successfully enriched node data from Cassandra.")
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

def save_maps_in_parallel(node_id_map: DataFrame, edge_map_df: DataFrame, temp_path: str):
    """
    Saves node and edge mapping data to a temporary HDFS location in a distributed
    manner (as CSVs) to avoid collecting all data on the driver.

    Args:
        node_id_map (DataFrame): DataFrame with ["global_node_id", "partition_id"].
        edge_map_df (DataFrame): DataFrame with ["global_edge_id", "partition_id"].
        temp_path (str): The HDFS path to save the intermediate part-files.
    """
    logger.info(f"Saving node and edge maps in parallel to temporary location: {temp_path}")
    
    # Spark will write multiple part-files in each directory, which is scalable.
    node_id_map.write.mode("overwrite").parquet(os.path.join(temp_path, "node_map_parts"))
    edge_map_df.write.mode("overwrite").parquet(os.path.join(temp_path, "edge_map_parts"))


def assemble_maps_from_parts(spark: SparkSession, temp_path: str, output_path: str,
                             total_num_nodes: int, total_num_edges: int, config: Dict[str, Any]):
    """
    Assembles the final node/edge map tensors from the parallel-saved part-files.
    This function runs on the driver and collects only the necessary mapping data.

    Args:
        spark (SparkSession): The Spark session.
        temp_path (str): The HDFS path where intermediate part-files are stored.
        output_path (str): The final HDFS output path for the .pt files.
        total_num_nodes (int): The total number of nodes in the graph.
        total_num_edges (int): The total number of edges in the graph.
        config (Dict[str, Any]): The application configuration.
    """
    logger.info("Assembling final map files from parallel parts...")
    
    # --- Assemble Node Map ---
    node_map_parts_df = spark.read.parquet(os.path.join(temp_path, "node_map_parts"))
    node_map_pd = node_map_parts_df.toPandas()
    
    node_map_tensor = torch.full((total_num_nodes,), -1, dtype=torch.long)
    node_map_tensor[node_map_pd['global_node_id'].values] = torch.tensor(node_map_pd['partition_id'].values, dtype=torch.long)

    # --- Assemble Edge Map ---
    edge_map_parts_df = spark.read.parquet(os.path.join(temp_path, "edge_map_parts"))
    edge_map_pd = edge_map_parts_df.toPandas()

    edge_map_tensor = torch.full((total_num_edges,), -1, dtype=torch.long)
    edge_map_tensor[edge_map_pd['global_edge_id'].values] = torch.tensor(edge_map_pd['partition_id'].values, dtype=torch.long)

    # --- Save Final Tensors ---
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save node_map.pt
        node_map_path = os.path.join(tmpdir, "node_map.pt")
        torch.save(node_map_tensor, node_map_path)
        _write_to_hdfs_from_driver(node_map_path, f"{output_path}/node_map.pt", config)

        # Save edge_map.pt
        edge_map_path = os.path.join(tmpdir, "edge_map.pt")
        torch.save(edge_map_tensor, edge_map_path)
        _write_to_hdfs_from_driver(edge_map_path, f"{output_path}/edge_map.pt", config)
    
    logger.info("Successfully assembled and saved final map files.")
    
    # --- Clean up temporary directory ---
    try:
        logger.info(f"Cleaning up temporary map directory: {temp_path}")
        # Use the hdfs library client to remove the temporary directory
        namenode_host = config["hdfs_base_uri"].split('//')[1].split(':')[0]
        webhdfs_url = f"http://{namenode_host}:50070"
        client = InsecureClient(webhdfs_url, user=config.get("hdfs_user", "root"))
        client.delete(temp_path, recursive=True)
    except Exception as e:
        logger.warning(f"Could not clean up temporary directory {temp_path}. Please remove it manually. Error: {e}")

def save_partition_worker(partition_index: int, iterator: iter, config: Dict[str, Any], total_num_nodes: int):
    """
    Worker function executed via `mapPartitionsWithIndex`. It processes an
    iterator of rows for a single Spark partition, creates PyG-compatible files,
    and uploads them directly to HDFS.

    Args:
        partition_index (int): The physical index of the Spark RDD partition.
                               Note: This may not be the same as the logical graph partition ID.
        iterator (iter): An iterator over the Spark Rows for this partition.
        config (Dict[str, Any]): The broadcasted application configuration.
        total_num_nodes (int): The broadcasted total number of nodes in the graph.
    """
    # Import necessary libraries within the worker, as this code runs on executors.
    import pandas as pd
    import numpy as np
    import torch
    import os
    import tempfile
    import traceback
    from collections import defaultdict
    from hdfs import InsecureClient

    # Consume the iterator to materialize the rows for this partition into a list.
    rows = list(iterator)
    if not rows:
        # If the partition is empty, there's nothing to do.
        return iter([])

    # Since we repartitioned by 'partition_id', all rows in this iterator should belong
    # to the same logical graph partition. We can safely get it from the first row.
    logical_partition_id = rows[0]['partition_id']

    try:
        # Extract configuration needed for this worker.
        output_path = config["output_path"]
        namenode_host = config["hdfs_base_uri"].split('//')[1].split(':')[0]
        webhdfs_url = f"http://{namenode_host}:50070"
        hdfs_user = config.get("hdfs_user", "root")
        client = InsecureClient(webhdfs_url, user=hdfs_user)

        # Convert list of Spark Rows to a Pandas DataFrame for easier manipulation.
        pdf = pd.DataFrame([row.asDict() for row in rows])

        with tempfile.TemporaryDirectory() as tmpdir:
            # --- 1. Process Nodes and Create `node_feats.pt` ---
            nodes_pd = pdf[pdf['type'] == 'node'].sort_values('global_node_id').reset_index(drop=True)
            
            # This dictionary structure matches the PyG Partitioner output for homogeneous graphs.
            node_feats_dict = {
                'global_id': torch.from_numpy(nodes_pd['global_node_id'].to_numpy(dtype=np.int64)),
                'feats': {
                    'x': torch.from_numpy(np.array(nodes_pd['features'].tolist(), dtype=np.float32)),
                    'y': torch.from_numpy(nodes_pd['label'].to_numpy(dtype=np.int64)).to(torch.long),
                    'train_mask': torch.from_numpy(nodes_pd['train_mask'].to_numpy(dtype=np.bool_)),
                    'val_mask': torch.from_numpy(nodes_pd['val_mask'].to_numpy(dtype=np.bool_)),
                    'test_mask': torch.from_numpy(nodes_pd['test_mask'].to_numpy(dtype=np.bool_)),
                }
            }
            node_feats_path = os.path.join(tmpdir, "node_feats.pt")
            torch.save(node_feats_dict, node_feats_path)

            # --- 2. Process Edges and Create `graph.pt` ---
            edges_pd = pdf[pdf['type'] == 'edge']
            
            # IMPORTANT: Sort edges by the destination node ID ('dst_global') to mimic
            # PyG's `sort_csc` behavior, which is required for efficient sampling.
            edges_pd = edges_pd.sort_values('dst_global').reset_index(drop=True)
            
            if not edges_pd.empty:
                # This dictionary structure matches the PyG Partitioner output for homogeneous graphs.
                graph_dict = {
                    'row': torch.from_numpy(edges_pd['src_global'].to_numpy(dtype=np.int64)),
                    'col': torch.from_numpy(edges_pd['dst_global'].to_numpy(dtype=np.int64)),
                    'edge_id': torch.from_numpy(edges_pd['global_edge_id'].to_numpy(dtype=np.int64)),
                    'size': (total_num_nodes, total_num_nodes),
                }
            else:
                # Handle cases where a partition might have no internal edges.
                graph_dict = {
                    'row': torch.empty(0, dtype=torch.long),
                    'col': torch.empty(0, dtype=torch.long),
                    'edge_id': torch.empty(0, dtype=torch.long),
                    'size': (total_num_nodes, total_num_nodes),
                }
            graph_path = os.path.join(tmpdir, "graph.pt")
            torch.save(graph_dict, graph_path)

            # --- 3. Create `edge_feats.pt` (empty for now) ---
            # This matches the PyG Partitioner's behavior when no edge features are present.
            edge_feats_dict = defaultdict()
            edge_feats_path = os.path.join(tmpdir, "edge_feats.pt")
            torch.save(edge_feats_dict, edge_feats_path)
            
            # --- 4. Upload all files for this partition to HDFS ---
            hdfs_partition_dir = f"{output_path}/part_{logical_partition_id}"
            client.upload(f"{hdfs_partition_dir}/node_feats.pt", node_feats_path, overwrite=True)
            client.upload(f"{hdfs_partition_dir}/graph.pt", graph_path, overwrite=True)
            client.upload(f"{hdfs_partition_dir}/edge_feats.pt", edge_feats_path, overwrite=True)
            
        # Yield a success status record for this partition.
        yield (logical_partition_id, True, "Success")

    except Exception:
        # If any error occurs, yield a failure record with the traceback for debugging.
        error_msg = f"ERROR processing partition for logical_id={logical_partition_id}:\n{traceback.format_exc()}"
        print(error_msg) # This will print to the executor's log.
        yield (logical_partition_id, False, error_msg)

def save_partitions_for_pyg(spark: SparkSession, enriched_nodes_df: DataFrame, edges_df: DataFrame,
                           total_num_nodes: int, config: Dict[str, Any]):
    """
    Saves the partitioned graph data in a PyG-compatible format using a
    memory-efficient RDD.mapPartitions approach. This is compatible with
    Spark 2.4 and does not use PyArrow for data serialization.

    Args:
        spark (SparkSession): The active Spark session.
        enriched_nodes_df (DataFrame): DataFrame of nodes with features, labels, and partition assignments.
        edges_df (DataFrame): DataFrame of edges with global source/destination IDs.
        total_num_nodes (int): The total number of nodes in the graph.
        config (Dict[str, Any]): The application configuration.
    """
    logger.info("Starting distributed save of partition data in PyG-compatible format...")
    output_path = config["output_path"]

    # --- 1. Save `META.json` from the driver ---
    # This structure matches the one created by the PyG Partitioner for homogeneous graphs.
    meta = {
        'num_parts': config["partitions"],
        'is_hetero': False,
        'node_types': None,
        'edge_types': None,
        'is_sorted': True,  # We ensure this by sorting by destination node in the worker.
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        meta_path = os.path.join(tmpdir, "META.json")
        with open(meta_path, 'w') as f:
            json.dump(meta, f)
        _write_to_hdfs_from_driver(meta_path, f"{output_path}/META.json", config)

    # --- 2. Prepare a unified DataFrame for nodes and edges ---
    # This mirrors the logic from your original script.
    intra_partition_edges = (
        edges_df.join(enriched_nodes_df.select("global_node_id", "partition_id").alias("src_map"), F.col("src_global") == F.col("src_map.global_node_id"))
        .join(enriched_nodes_df.select("global_node_id", "partition_id").alias("dst_map"), F.col("dst_global") == F.col("dst_map.global_node_id"))
        .where(F.col("src_map.partition_id") == F.col("dst_map.partition_id"))
        .select(F.col("src_map.partition_id").alias("partition_id"), "src_global", "dst_global", "global_edge_id")
    )
    nodes_for_grouping = enriched_nodes_df.select(
        "partition_id", F.lit("node").alias("type"), "global_node_id", "features", "label", "train_mask", "val_mask", "test_mask",
        F.lit(None).cast(LongType()).alias("src_global"), F.lit(None).cast(LongType()).alias("dst_global"), F.lit(None).cast(LongType()).alias("global_edge_id")
    )
    edges_for_grouping = intra_partition_edges.select(
        "partition_id", F.lit("edge").alias("type"), F.lit(None).alias("global_node_id"),
        F.lit(None).cast(ArrayType(FloatType())).alias("features"), F.lit(None).cast(IntegerType()).alias("label"),
        F.lit(None).cast(BooleanType()).alias("train_mask"), F.lit(None).cast(BooleanType()).alias("val_mask"), F.lit(None).cast(BooleanType()).alias("test_mask"),
        "src_global", "dst_global", "global_edge_id"
    )
    unified_data_df = nodes_for_grouping.unionByName(edges_for_grouping)

    # --- 3. Repartition and Execute Processing via RDD API ---
    # Repartition by our logical graph 'partition_id'. This is crucial to ensure all data
    # for a single PyG partition (e.g., part_0) ends up in the same Spark RDD partition.
    data_to_save_rdd = unified_data_df.repartition(config["partitions"], "partition_id").rdd

    # Broadcast large, read-only variables to all executors for efficiency.
    b_config = spark.sparkContext.broadcast(config)
    b_total_num_nodes = spark.sparkContext.broadcast(total_num_nodes)

    # Define the schema for the results we expect back from our worker function.
    result_schema = StructType([
        StructField("partition_id", IntegerType(), True),
        StructField("success", BooleanType(), False),
        StructField("message", StringType(), False)
    ])
    
    logger.info("Executing save operation for all partitions using RDD.mapPartitionsWithIndex...")
    
    # Use a lambda function to pass the broadcasted variables to our worker.
    result_rdd = data_to_save_rdd.mapPartitionsWithIndex(
        lambda idx, iterator: save_partition_worker(idx, iterator, b_config.value, b_total_num_nodes.value)
    )
    
    # Convert the result RDD back to a DataFrame to check for failures.
    result_df = result_rdd.toDF(result_schema).coalesce(1)
    failed_partitions = result_df.where(F.col("success") == False).collect()
    
    if failed_partitions:
        logger.error(f"{len(failed_partitions)} partitions failed to save. Check executor logs for detailed tracebacks.")
        for row in failed_partitions:
            logger.error(f"Partition [{row['partition_id']}] failed with message: {row['message']}")
    else:
        logger.info("All partitions processed and saved successfully.")

def main():
    """Main execution function with scalable map generation."""
    spark = None
    # Define a temporary path for intermediate map files
    temp_map_path = os.path.join(CONFIG["output_path"], "_tmp_maps")
    
    try:
        spark = create_spark_session(CONFIG)
        
        # --- Stage 1: Main Distributed Processing ---
        logger.info("--- STAGE 1: Starting Main Distributed Graph Processing ---")
        
        nodes_df, raw_edges_df = load_from_nebula(spark, CONFIG)
        
        # Scalable Node ID Generation
        node_id_map_rdd = nodes_df.select("id", "cust_id").distinct().rdd.map(lambda row: (row.id, row.cust_id)).zipWithIndex()
        node_id_map = node_id_map_rdd.map(lambda x: (x[0][0], x[0][1], x[1])).toDF(["id", "cust_id", "global_node_id"])

        nodes_df.cache()
        node_id_map.cache()

        # Scalable Edge ID Generation
        src_node_map = node_id_map.select("id", "global_node_id").withColumnRenamed("id", "src") .withColumnRenamed("global_node_id", "src_global")
        dst_node_map = node_id_map.select("id", "global_node_id").withColumnRenamed("id", "dst").withColumnRenamed("global_node_id", "dst_global")
        edges_with_global_nodes = raw_edges_df.join(src_node_map, "src", "inner").join(dst_node_map, "dst", "inner")
        edge_id_map_rdd = edges_with_global_nodes.select("src_global", "dst_global").rdd.zipWithIndex()
        edge_id_map_df = edge_id_map_rdd.map(lambda x: (x[0][0], x[0][1], x[1])).toDF(["src_global", "dst_global", "global_edge_id"])
        edges_df = edges_with_global_nodes.join(edge_id_map_df, ["src_global", "dst_global"], "inner")
        
        edges_df.cache()
        total_num_nodes = node_id_map.count()
        total_num_edges = edges_df.count()
        logger.info(f"Total nodes: {total_num_nodes}, Total edges: {total_num_edges}")

        partitioned_nodes_df = partition_graph(nodes_df, raw_edges_df, CONFIG["partitions"])
        
        partitioned_nodes_with_global_id = partitioned_nodes_df.join(node_id_map, "id", "inner")
        nodes_df.unpersist()

        enriched_nodes_df = enrich_data_from_cassandra(spark, partitioned_nodes_with_global_id, CONFIG)

        # --- DIAGNOSTIC ---
        logger.info("Counting nodes that did NOT have features in Cassandra, by partition:")
        enriched_nodes_df.where(F.col("features").isNull()) \
            .groupBy("partition_id") \
            .count() \
            .orderBy("partition_id") \
            .show(200) # Show more rows to see all partitions
        # ------------------

        final_nodes_df = add_temporal_train_test_split(enriched_nodes_df)
        final_nodes_df.cache()
        print(f"DEBUG: Count of final_nodes_df before map creation: {final_nodes_df.count()}")

        # Create the mapping dataframes needed for the parallel save
        node_map_for_save = final_nodes_df.select("global_node_id", "partition_id")
        print(f"DEBUG: Count of node_map_for_save: {node_map_for_save.count()}")

        # edge_map_for_save = edges_df.join(final_nodes_df.select("id", "partition_id"), edges_df.src == final_nodes_df.id, "inner") \
        #                             .select("global_edge_id", "partition_id")

        # --- DETAILED EDGE DIAGNOSTICS ---
        # Use a LEFT join to find edges whose source nodes are missing from the final node list
        edge_join_df = edges_df.join(
            final_nodes_df.select("id", "partition_id"),
            edges_df.src == final_nodes_df.id,
            "left"  # Use a LEFT join for diagnostics
        )

        # Count how many edges failed to find a source node partition
        missing_src_nodes_count = edge_join_df.where(F.col("partition_id").isNull()).count()
        print(f"CRITICAL: Found {missing_src_nodes_count} edges whose source node was not in the final node list.")

        # This is the original failing line
        edge_map_for_save = edges_df.join(
            final_nodes_df.select("id", "partition_id"),
            edges_df.src == final_nodes_df.id,
            "inner"
        ).select("global_edge_id", "partition_id")
        print(f"DEBUG: Count of edge_map_for_save: {edge_map_for_save.count()}")

        # Add an assertion to fail fast
        if node_map_for_save.count() == 0 or edge_map_for_save.count() == 0:
            raise ValueError("One or both of the map DataFrames are empty. Halting before write.")

        # Execute the parallel saving operations
        save_maps_in_parallel(node_map_for_save, edge_map_for_save, temp_map_path)
        save_partitions_for_pyg(spark, final_nodes_df, edges_df, total_num_nodes, CONFIG)

        edges_df.unpersist()
        final_nodes_df.unpersist()
        node_id_map.unpersist()
        
        logger.info("--- STAGE 1: Main Distributed Processing Finished ---")

        # --- Stage 2: Final Assembly on Driver ---
        logger.info("--- STAGE 2: Starting Final Map Assembly on Driver ---")
        assemble_maps_from_parts(spark, temp_map_path, CONFIG["output_path"], total_num_nodes, total_num_edges, CONFIG)
        logger.info("--- STAGE 2: Final Map Assembly Finished ---")

        logger.info("GNN data preparation pipeline finished successfully.")

    except Exception as e:
        logger.error(f"An error occurred during the Spark job: {e}", exc_info=True)
        traceback.print_exc()
        sys.exit(1)
    finally:
        if spark:
            logger.info("Stopping Spark session.")
            spark.stop()

if __name__ == "__main__":
    main()
