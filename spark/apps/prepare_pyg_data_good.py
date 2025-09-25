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
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, LongType, IntegerType, FloatType, BooleanType
from pyspark.sql.functions import pandas_udf, PandasUDFType
from graphframes import GraphFrame

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration Section ---
CONFIG = {
    "app_name": "GNN_Data_Prep_Pipeline",
    "partitions": 3,
    "node_limit": 1000000,

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
        .config(f"spark.sql.shuffle.partitions", 200)
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "1000")
    )
    
    spark = spark_builder.getOrCreate()
    spark.sparkContext.setCheckpointDir(config["checkpoint_dir"])
    
    logger.info("Spark session created successfully.")
    return spark


def load_from_nebula(spark: SparkSession, config: Dict[str, Any]) -> (DataFrame, DataFrame):
    """Loads a limited subset of vertices and ALL edges from NebulaGraph."""
    logger.info(f"Creating plan to load a limit of {config['node_limit']} vertices...")
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
        .distinct()
        .limit(config['node_limit'])
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
    g = GraphFrame(nodes_df.select("id"), edges_df)
    partition_assignments = g.labelPropagation(maxIter=1)
    
    final_partitions_df = partition_assignments.withColumn(
        "partition_id", (F.col("label") % num_partitions).cast("int")
    ).select("id", "partition_id")

    logger.info("Graph partitioning complete. Partition distribution:")
    final_partitions_df.groupBy("partition_id").count().orderBy("partition_id").show()
    return final_partitions_df


def enrich_data_from_cassandra(spark: SparkSession, partitioned_nodes_df: DataFrame, config: Dict[str, Any]) -> DataFrame:
    """Enriches node data by joining with features from Cassandra via join pushdown."""
    logger.info(f"Enriching node data from Cassandra table: {config['cassandra']['keyspace']}.{config['cassandra']['table']}")
    features_df = (
        spark.read.format("org.apache.spark.sql.cassandra")
        .options(table=config['cassandra']['table'], keyspace=config['cassandra']['keyspace'])
        .load()
    )
    
    enriched_nodes_df = partitioned_nodes_df.join(
        features_df,
        "cust_id",
        "inner" 
    )

    if enriched_nodes_df.rdd.isEmpty():
        raise ValueError(
            "The INNER join on 'cust_id' between nodes from Nebula and features from Cassandra resulted in an empty DataFrame."
        )
    
    logger.info("Successfully enriched node data from Cassandra.")
    return enriched_nodes_df

def add_temporal_train_test_split(df: DataFrame, train_perc: float = 0.7, val_perc: float = 0.15) -> DataFrame:
    """Adds train/validation/test split masks based on a temporal column."""
    logger.info(f"Creating temporal splits: {train_perc*100}% train, {val_perc*100}% val...")
    df_with_unix_ts = df.withColumn("unix_ts", F.col("timestamp").cast("long"))
    split_points = df_with_unix_ts.approxQuantile("unix_ts", [train_perc, train_perc + val_perc], 0.01)
    train_end_ts, val_end_ts = split_points[0], split_points[1]
    df_with_masks = df_with_unix_ts.withColumn(
        "train_mask", F.col("unix_ts") <= train_end_ts
    ).withColumn(
        "val_mask", (F.col("unix_ts") > train_end_ts) & (F.col("unix_ts") <= val_end_ts)
    ).withColumn(
        "test_mask", F.col("unix_ts") > val_end_ts
    ).drop("unix_ts") 
    df_with_masks.groupBy("train_mask", "val_mask", "test_mask").count().show()
    return df_with_masks

def _write_to_hdfs_from_driver(local_path: str, hdfs_path: str, config: Dict[str, Any]):
    namenode_host = config["hdfs_base_uri"].split('//')[1].split(':')[0]
    webhdfs_url = f"http://{namenode_host}:50070"
    client = InsecureClient(webhdfs_url, user='root')
    try:
        client.upload(hdfs_path=hdfs_path, local_path=local_path, overwrite=True)
    except Exception as e:
        logger.error(f"Failed to upload to HDFS via WebHDFS. Error: {e}")
        raise e

def generate_and_save_maps_on_driver(final_nodes_df: DataFrame, edges_df: DataFrame, total_num_nodes: int, total_num_edges: int, config: Dict[str, Any]):
    """
    Collects node and edge mapping data to the driver, assembles the PyG map
    tensors, and saves them to HDFS.

    WARNING: This function collects data to the driver. It is suitable for graphs
    where the node and edge counts are manageable for the driver's memory, but
    it can cause OutOfMemory errors on the driver for very large graphs.
    """
    logger.info("Collecting node and edge mapping data to the driver...")
    
    # --- Collect Node Map Data ---
    node_map_pd = final_nodes_df.select("global_node_id", "partition_id").toPandas()
    
    # --- Collect Edge Map Data ---
    edge_map_df = edges_df.join(
        final_nodes_df.select("global_node_id", "partition_id").alias("src_nodes"),
        edges_df.src_global == F.col("src_nodes.global_node_id"),
        "inner"
    ).select("global_edge_id", "partition_id")
    edge_map_pd = edge_map_df.toPandas()
    
    logger.info(f"Successfully collected {len(node_map_pd)} node mappings and {len(edge_map_pd)} edge mappings.")

    # --- Assemble and Save Tensors ---
    logger.info("Assembling final map files on the driver...")
    node_map_tensor = torch.full((total_num_nodes,), -1, dtype=torch.long)
    node_map_tensor[node_map_pd['global_node_id'].values] = torch.tensor(node_map_pd['partition_id'].values, dtype=torch.long)

    edge_map_tensor = torch.full((total_num_edges,), -1, dtype=torch.long)
    edge_map_tensor[edge_map_pd['global_edge_id'].values] = torch.tensor(edge_map_pd['partition_id'].values, dtype=torch.long)

    output_path = config["output_path"]
    with tempfile.TemporaryDirectory() as tmpdir:
        node_map_path = os.path.join(tmpdir, "node_map.pt")
        torch.save(node_map_tensor, node_map_path)
        _write_to_hdfs_from_driver(node_map_path, f"{output_path}/node_map.pt", config)

        edge_map_path = os.path.join(tmpdir, "edge_map.pt")
        torch.save(edge_map_tensor, edge_map_path)
        _write_to_hdfs_from_driver(edge_map_path, f"{output_path}/edge_map.pt", config)
    
    logger.info("Successfully assembled and saved final map files to HDFS.")

def save_partitioned_data(spark: SparkSession, enriched_nodes_df: DataFrame, edges_df: DataFrame, total_num_nodes: int, total_num_edges: int, num_classes: int, config: Dict[str, Any]):
    """
    Saves the partitioned graph data in the format expected by PyG's distributed loader.
    """
    logger.info("Starting distributed save of partition data...")
    output_path = config["output_path"]
    node_feature_dim = 600  # The specified node feature dimension

    # --- 1. Save Global META.json from Driver ---
    meta = {
        'is_sorted': False,
        'is_hetero': False,
        'num_nodes': total_num_nodes,
        'num_edges': total_num_edges,
        'num_parts': config["partitions"],
        'num_classes': num_classes,
        'node_feat_schema': {"__feat__": node_feature_dim}
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        meta_path = os.path.join(tmpdir, "META.json")
        with open(meta_path, 'w') as f:
            json.dump(meta, f)
        _write_to_hdfs_from_driver(meta_path, f"{output_path}/META.json", config)
    logger.info(f"Successfully saved META.json to HDFS.")

    # --- 2. Assemble and Save Global labels.pt from Driver ---
    logger.info("Assembling and saving global labels and masks tensor on the driver...")
    try:
        # Collect all node data needed for labels.pt, ordered by global_node_id for correct indexing
        labels_pd = enriched_nodes_df.select("global_node_id", "label", "train_mask", "val_mask", "test_mask") \
                                     .orderBy("global_node_id") \
                                     .toPandas()

        # Create the dictionary of tensors
        labels_data = {
            'y': torch.from_numpy(labels_pd['label'].to_numpy(dtype=np.int64)),
            'train_mask': torch.from_numpy(labels_pd['train_mask'].to_numpy(dtype=np.bool_)),
            'val_mask': torch.from_numpy(labels_pd['val_mask'].to_numpy(dtype=np.bool_)),
            'test_mask': torch.from_numpy(labels_pd['test_mask'].to_numpy(dtype=np.bool_)),
        }
        
        # Verify shape consistency
        assert labels_data['y'].shape[0] == total_num_nodes, "Shape mismatch for labels tensor."

        # Save to HDFS
        with tempfile.TemporaryDirectory() as tmpdir:
            labels_path = os.path.join(tmpdir, "labels.pt")
            torch.save(labels_data, labels_path)
            _write_to_hdfs_from_driver(labels_path, f"{output_path}/labels.pt", config)
        logger.info(f"Successfully saved labels.pt for {total_num_nodes} nodes to HDFS.")
    except Exception as e:
        logger.error(f"Failed to create and save labels.pt on the driver. Error: {e}", exc_info=True)
        raise e

    # --- 3. Prepare DataFrames for Distributed Partition Saving ---
    intra_partition_edges = (
        edges_df.join(enriched_nodes_df.alias("src_map"), F.col("src_global") == F.col("src_map.global_node_id"))
        .join(enriched_nodes_df.alias("dst_map"), F.col("dst_global") == F.col("dst_map.global_node_id"))
        .where(F.col("src_map.partition_id") == F.col("dst_map.partition_id"))
        .select(F.col("src_map.partition_id").alias("partition_id"), "src_global", "dst_global", "global_edge_id")
    )
    nodes_for_grouping = enriched_nodes_df.select("partition_id", F.lit("node").alias("type"), "global_node_id", "features", F.lit(None).cast(LongType()).alias("src_global"), F.lit(None).cast(LongType()).alias("dst_global"), F.lit(None).cast(LongType()).alias("global_edge_id"))
    edges_for_grouping = intra_partition_edges.select("partition_id", F.lit("edge").alias("type"), F.lit(None).alias("global_node_id"), F.lit(None).cast(ArrayType(FloatType())).alias("features"), "src_global", "dst_global", "global_edge_id")
    grouped_data = nodes_for_grouping.unionByName(edges_for_grouping)
    
    # Broadcast variables for UDF
    b_total_num_nodes = spark.sparkContext.broadcast(total_num_nodes)
    namenode_host = config["hdfs_base_uri"].split('//')[1].split(':')[0]
    b_webhdfs_url = spark.sparkContext.broadcast(f"http://{namenode_host}:50070")
    b_output_path = spark.sparkContext.broadcast(output_path)
    b_hdfs_user = spark.sparkContext.broadcast(config.get("hdfs_user", "root"))
    result_schema = StructType([StructField("partition_id", IntegerType()), StructField("success", BooleanType())])

    # --- 4. Pandas UDF for Saving Each Partition's Data ---
    @pandas_udf(result_schema, PandasUDFType.GROUPED_MAP)
    def process_and_save_partition_fn(pdf: pd.DataFrame) -> pd.DataFrame:
        if pdf.empty: return pd.DataFrame({'partition_id': [], 'success': []})
        partition_id = int(pdf['partition_id'].iloc[0])
        success = False
        try:
            client = InsecureClient(b_webhdfs_url.value, user=b_hdfs_user.value)
            with tempfile.TemporaryDirectory() as tmpdir:
                # Node processing
                nodes_pd = pdf[pdf['type'] == 'node'].sort_values('global_node_id').reset_index()
                
                # Standardize feature vectors
                processed_features = []
                for features_list in nodes_pd['features']:
                    if features_list is None or not isinstance(features_list, list):
                        processed_features.append(np.zeros(node_feature_dim, dtype=np.float32))
                    else:
                        vec = np.array(features_list, dtype=np.float32)
                        if vec.size < node_feature_dim: vec = np.pad(vec, (0, node_feature_dim - vec.size), 'constant')
                        elif vec.size > node_feature_dim: vec = vec[:node_feature_dim]
                        processed_features.append(vec)
                
                feature_array = np.vstack(processed_features) if processed_features else np.empty((0, node_feature_dim), dtype=np.float32)

                # --- Modified: Create the requested structure for node_feats.pt ---
                feats_dict = {
                    'x': torch.from_numpy(feature_array)
                }
                node_feats = {
                    'global_id': torch.tensor(nodes_pd['global_node_id'].values, dtype=torch.long),
                    'feats': feats_dict
                }
                node_feats_path = os.path.join(tmpdir, "node_feats.pt")
                torch.save(node_feats, node_feats_path)
                
                # Edge processing and graph.pt creation
                edges_pd = pdf[pdf['type'] == 'edge']
                # --- Modified: Removed redundant 'node_id' from graph.pt ---
                graph_dict = {
                    'size': (b_total_num_nodes.value, b_total_num_nodes.value),
                    'row': torch.tensor(edges_pd['src_global'].values, dtype=torch.long),
                    'col': torch.tensor(edges_pd['dst_global'].values, dtype=torch.long),
                    'edge_id': torch.tensor(edges_pd['global_edge_id'].values, dtype=torch.long)
                }
                graph_path = os.path.join(tmpdir, "graph.pt")
                torch.save(graph_dict, graph_path)
                
                # Save empty edge_feats.pt
                edge_feats_path = os.path.join(tmpdir, "edge_feats.pt")
                torch.save(defaultdict(), edge_feats_path)
                
                # Upload files to HDFS
                hdfs_partition_dir = f"{b_output_path.value}/part_{partition_id}"
                client.upload(f"{hdfs_partition_dir}/node_feats.pt", node_feats_path, overwrite=True)
                client.upload(f"{hdfs_partition_dir}/graph.pt", graph_path, overwrite=True)
                client.upload(f"{hdfs_partition_dir}/edge_feats.pt", edge_feats_path, overwrite=True)
            success = True
        except Exception as e:
            print(f"--- ERROR IN PARTITION {partition_id} ---")
            traceback.print_exc()
            success = False
        return pd.DataFrame([{'partition_id': partition_id, 'success': success}])

    # --- 5. Execute and Verify ---
    result = grouped_data.groupBy("partition_id").apply(process_and_save_partition_fn)
    failed_partitions = result.where(F.col("success") == False).count()
    if failed_partitions > 0: logger.error(f"{failed_partitions} partitions failed to save. Check executor logs.")
    else: logger.info("All partitions processed and saved successfully.")

def main():
    """
    Main execution function refactored to the more efficient "enrich-first" workflow.
    This eliminates wasted computation and the need for a complex remapping step.
    """
    spark = None
    try:
        spark = create_spark_session(CONFIG)
        
        # --- Stage 1: Load, Enrich, and Establish Final Node Set ---
        logger.info("Loading initial node set from Nebula...")
        initial_nodes_df, all_edges_df = load_from_nebula(spark, CONFIG)

        logger.info("Enriching nodes with features from Cassandra to get the final, valid node set...")
        enriched_nodes_df = enrich_data_from_cassandra(spark, initial_nodes_df, CONFIG)
        enriched_nodes_df.cache()
        
        final_node_count = enriched_nodes_df.count()
        logger.info(f"COUNT: Final number of valid nodes with features: {final_node_count}")

        # --- Stage 2: Filter Edges and Generate Final, Dense IDs ---
        logger.info("Filtering edge list to only include edges between valid nodes...")
        valid_node_ids = enriched_nodes_df.select("id").distinct()
        
        final_edges_subset_df = all_edges_df \
            .join(valid_node_ids.withColumnRenamed("id", "src"), "src", "inner") \
            .join(valid_node_ids.withColumnRenamed("id", "dst"), "dst", "inner")

        logger.info("Generating final, dense, 0-based IDs for nodes and edges...")
        
        node_id_map_rdd = enriched_nodes_df.select("id", "cust_id").distinct().rdd.map(lambda r: (r.id, r.cust_id)).zipWithIndex()
        final_node_id_map = node_id_map_rdd.map(lambda x: (x[0][0], x[0][1], x[1])).toDF(["id", "cust_id", "global_node_id"])
        final_node_id_map.cache()

        src_map = final_node_id_map.select("id", "global_node_id").withColumnRenamed("id", "src").withColumnRenamed("global_node_id", "src_global")
        dst_map = final_node_id_map.select("id", "global_node_id").withColumnRenamed("id", "dst").withColumnRenamed("global_node_id", "dst_global")
        edges_w_dense_node_ids = final_edges_subset_df.join(src_map, "src", "inner").join(dst_map, "dst", "inner")

        distinct_edges = edges_w_dense_node_ids.select("src_global", "dst_global").distinct()
        edge_id_rdd = distinct_edges.rdd.zipWithIndex()
        final_edges_df = edge_id_rdd.map(lambda x: (x[0][0], x[0][1], x[1])).toDF(["src_global", "dst_global", "global_edge_id"])
        final_edges_df.cache()

        total_num_nodes = final_node_count
        total_num_edges = final_edges_df.count()
        logger.info(f"COUNT (Final): Total nodes for training: {total_num_nodes}")
        logger.info(f"COUNT (Final): Total unique edges for training: {total_num_edges}")

        # --- Stage 3: Partitioning and Final Data Preparation ---
        nodes_with_ids = enriched_nodes_df.join(final_node_id_map, ["id", "cust_id"], "inner")

        logger.info("Partitioning the final, valid graph...")
        partitioned_nodes_df = partition_graph(nodes_with_ids, final_edges_subset_df, CONFIG["partitions"])
        
        nodes_with_partitions = nodes_with_ids.join(partitioned_nodes_df, "id", "inner")
        
        final_nodes_df = add_temporal_train_test_split(nodes_with_partitions)
        final_nodes_df.cache()

        # New: Calculate num_classes for metadata
        num_classes = final_nodes_df.select("label").distinct().count()
        logger.info(f"Discovered {num_classes} distinct classes for the prediction task.")

        # --- Stage 4: Saving ---
        logger.info("Starting distributed save of partitioned data files...")
        save_partitioned_data(spark, final_nodes_df, final_edges_df, total_num_nodes, total_num_edges, num_classes, CONFIG)
        
        logger.info("Starting generation of global map files on the driver...")
        generate_and_save_maps_on_driver(final_nodes_df, final_edges_df, total_num_nodes, total_num_edges, CONFIG)
        
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
