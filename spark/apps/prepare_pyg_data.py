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

def enrich_data_from_cassandra_2(spark: SparkSession, partitioned_nodes_df: DataFrame, config: Dict[str, Any]) -> DataFrame:
    """
    Enriches node data by joining with features from Cassandra.
    Includes a critical cleaning step to remove null characters from the join key.
    """
    logger.info(f"Enriching node data from Cassandra table: {config['cassandra']['keyspace']}.{config['cassandra']['table']}")
    
    # Clean the cust_id column by replacing the unicode null character ('\u0000') with an empty string.
    # This ensures the keys sent to Cassandra are clean and will match the TEXT PRIMARY KEY.
    nodes_with_clean_keys = partitioned_nodes_df.withColumn(
        "cust_id_clean", F.regexp_replace(F.col("cust_id"), "\\u0000", "")
    )

    features_df = (
        spark.read.format("org.apache.spark.sql.cassandra")
        .options(table=config['cassandra']['table'], keyspace=config['cassandra']['keyspace'])
        .load()
    )
    
    # Use a 'left' join to keep all nodes from Nebula, joining on the newly cleaned key.
    enriched_nodes_df = nodes_with_clean_keys.join(
        features_df,
        nodes_with_clean_keys.cust_id_clean == features_df.cust_id,
        "left"
    ).drop("cust_id_clean") # Drop the temporary clean key after the join is done
    
    logger.info("Successfully enriched node data from Cassandra.")
    return enriched_nodes_df

def enrich_data_from_cassandra_3(spark: SparkSession, partitioned_nodes_df: DataFrame, config: Dict[str, Any]) -> DataFrame:
    """
    Enriches node data using a performant, pushdown-capable join with Cassandra.
    This version avoids full table scans and cleans the join keys to ensure correctness.
    """
    logger.info("Enriching node data using a performant left join with Cassandra...")

    # 1. Clean the keys in the source DataFrame. This is the fix for the data integrity issue.
    # The regex '[^\p{Print}]' removes all non-printable characters.
    nodes_with_clean_keys = partitioned_nodes_df.withColumn(
        "join_key", F.regexp_replace(F.col("cust_id"), "[^\\p{Print}]", "")
    )

    # 2. Reference the Cassandra table directly as a Spark DataFrame.
    # Do NOT call .load() here. This keeps it as a reference for the pushdown.
    features_df = spark.read.format("org.apache.spark.sql.cassandra") \
        .options(table=config['cassandra']['table'], keyspace=config['cassandra']['keyspace']) \
        .load()

    # 3. Perform the join. The Spark-Cassandra connector is smart enough to see
    # this pattern and will "push down" the join, converting it into an efficient
    # `WHERE cust_id IN (...)` query instead of a full table scan.
    enriched_nodes_df = nodes_with_clean_keys.join(
        features_df,
        nodes_with_clean_keys.join_key == features_df.cust_id,
        "left"
    ).drop("join_key", "cust_id").withColumnRenamed("cust_id", "cust_id")

    logger.info("Successfully enriched node data from Cassandra.")
    return enriched_nodes_df

def enrich_data_from_cassandra_4(spark: SparkSession, partitioned_nodes_df: DataFrame, config: Dict[str, Any]) -> DataFrame:
    """
    Enriches node data by joining with features from Cassandra. If features are
    missing, generates them randomly.
    """
    logger.info(f"Enriching node data from Cassandra table: {config['cassandra']['keyspace']}.{config['cassandra']['table']}")
    features_df = (
        spark.read.format("org.apache.spark.sql.cassandra")
        .options(table=config['cassandra']['table'], keyspace=config['cassandra']['keyspace'])
        .load()
    )
    
    # Use a LEFT join to keep all nodes from Nebula, even if they don't have features
    enriched_df = partitioned_nodes_df.join(features_df, "cust_id", "left")
    
    # Check how many nodes are missing features
    missing_count = enriched_df.where(F.col("features").isNull()).count()
    
    if missing_count > 0:
        logger.info(f"Found {missing_count} nodes missing from Cassandra. Generating random data for them.")
        
        feature_dim = config.get("feature_dim", 600)
        b_feature_dim = spark.sparkContext.broadcast(feature_dim)

        # Define a UDF to generate random feature vectors based on Xavier initialization
        @F.udf(returnType=ArrayType(FloatType()))
        def generate_random_features_udf():
            import numpy as np
            # For a feature vector (no fan_out), a common Xavier uniform implementation
            # uses a limit of sqrt(3 / fan_in) to achieve a variance of 1 / fan_in.
            limit = np.sqrt(3.0 / b_feature_dim.value)
            return np.random.uniform(-limit, limit, size=b_feature_dim.value).tolist()

        # Define expressions for random labels and timestamps
        random_label_expr = (F.rand() * 2).cast(IntegerType())
        
        now_ts = F.current_timestamp().cast(LongType())
        one_year_in_seconds = 365 * 24 * 3600
        one_year_ago_ts = now_ts - one_year_in_seconds
        random_timestamp_expr = F.from_unixtime(one_year_ago_ts + (F.rand() * one_year_in_seconds))

        # Fill nulls with generated data
        final_df = enriched_df.withColumn(
            "features",
            F.when(F.col("features").isNull(), generate_random_features_udf()).otherwise(F.col("features"))
        ).withColumn(
            "label",
            F.when(F.col("label").isNull(), random_label_expr).otherwise(F.col("label"))
        ).withColumn(
            "timestamp",
            F.when(F.col("timestamp").isNull(), random_timestamp_expr).otherwise(F.col("timestamp"))
        )
    else:
        logger.info("No nodes were missing from Cassandra. No data generation needed.")
        final_df = enriched_df

    # Ensure final schema is consistent for all rows
    final_df = final_df.withColumn("label", F.col("label").cast(IntegerType())) \
                       .withColumn("timestamp", F.col("timestamp").cast("timestamp"))

    return final_df

def enrich_data_from_cassandra_5(spark: SparkSession, partitioned_nodes_df: DataFrame, config: Dict[str, Any]) -> DataFrame:
    """
    Enriches node data by joining with Cassandra. It first generates a default
    set of fake features for all nodes, then overwrites them with real features
    if a match is found in Cassandra.
    """
    logger.info("Generating baseline fake features for all nodes...")

    feature_dim = config.get("feature_dim", 600)
    b_feature_dim = spark.sparkContext.broadcast(feature_dim)

    # 1. Define UDF and expressions for fake data generation
    @F.udf(returnType=ArrayType(FloatType()))
    def generate_random_features_udf():
        import numpy as np
        # Xavier/Glorot uniform initialization
        limit = np.sqrt(3.0 / b_feature_dim.value)
        return np.random.uniform(-limit, limit, size=b_feature_dim.value).tolist()

    random_label_expr = (F.rand() * 2).cast(IntegerType())
    now_ts = F.current_timestamp().cast(LongType())
    one_year_in_seconds = 365 * 24 * 3600
    one_year_ago_ts = now_ts - one_year_in_seconds
    random_timestamp_expr = F.from_unixtime(one_year_ago_ts + (F.rand() * one_year_in_seconds))

    # 2. Add fake data columns to every node as a baseline
    nodes_with_fakes = partitioned_nodes_df.withColumn(
        "fake_features", generate_random_features_udf()
    ).withColumn(
        "fake_label", random_label_expr
    ).withColumn(
        "fake_timestamp", random_timestamp_expr
    )

    # 3. Load real features from Cassandra and rename columns to avoid ambiguity
    logger.info(f"Loading real features from Cassandra: {config['cassandra']['keyspace']}.{config['cassandra']['table']}")
    cassandra_features_df = (
        spark.read.format("org.apache.spark.sql.cassandra")
        .options(table=config['cassandra']['table'], keyspace=config['cassandra']['keyspace'])
        .load()
        .withColumnRenamed("features", "real_features")
        .withColumnRenamed("label", "real_label")
        .withColumnRenamed("timestamp", "real_timestamp")
    )

    # 4. Perform a LEFT join to bring in real features where they exist
    nodes_joined = nodes_with_fakes.join(
        cassandra_features_df,
        "cust_id",
        "left"
    )

    # 5. Coalesce the real and fake columns to create the final, complete columns
    # F.coalesce picks the first non-null value, effectively prioritizing real data.
    logger.info("Merging real features from Cassandra with generated fake features...")
    final_df = nodes_joined.withColumn(
        "features", F.coalesce(F.col("real_features"), F.col("fake_features"))
    ).withColumn(
        "label", F.coalesce(F.col("real_label"), F.col("fake_label"))
    ).withColumn(
        "timestamp", F.coalesce(F.col("real_timestamp"), F.col("fake_timestamp"))
    )

    # 6. Select the final set of columns required downstream
    final_df_selected = final_df.select(
        "id", "cust_id", "global_node_id", "partition_id", # Original node identifiers
        "features", "label", "timestamp" # Final merged columns
    )

    # Log the outcome for verification
    final_df_selected.cache()
    total_count = final_df_selected.count()
    real_count = final_df_selected.where(F.col("features") == F.col("real_features")).count() if "real_features" in final_df.columns else 0
    fake_count = total_count - real_count

    logger.info(f"Feature enrichment complete. Total nodes: {total_count}, "
                f"Nodes with real features from Cassandra: {real_count}, "
                f"Nodes with generated fake features: {fake_count}")

    return final_df_selected

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


def save_partitioned_data(spark: SparkSession, enriched_nodes_df: DataFrame, edges_df: DataFrame, total_num_nodes: int, config: Dict[str, Any]):
    """Saves the partitioned graph data (node features, graph structure) in a distributed manner."""
    logger.info("Starting distributed save of partition data (graph, features, etc.)...")
    output_path = config["output_path"]
    meta = {'num_parts': config["partitions"], 'is_hetero': False, 'node_types': None, 'edge_types': None}
    with tempfile.TemporaryDirectory() as tmpdir:
        meta_path = os.path.join(tmpdir, "META.json")
        with open(meta_path, 'w') as f: json.dump(meta, f)
        _write_to_hdfs_from_driver(meta_path, f"{output_path}/META.json", config)

    intra_partition_edges = (
        edges_df.join(enriched_nodes_df.alias("src_map"), F.col("src_global") == F.col("src_map.global_node_id"))
        .join(enriched_nodes_df.alias("dst_map"), F.col("dst_global") == F.col("dst_map.global_node_id"))
        .where(F.col("src_map.partition_id") == F.col("dst_map.partition_id"))
        .select(F.col("src_map.partition_id").alias("partition_id"), "src_global", "dst_global", "global_edge_id")
    )
    nodes_for_grouping = enriched_nodes_df.select("partition_id", F.lit("node").alias("type"), "global_node_id", "features", "label", "train_mask", "val_mask", "test_mask", F.lit(None).cast(LongType()).alias("src_global"), F.lit(None).cast(LongType()).alias("dst_global"), F.lit(None).cast(LongType()).alias("global_edge_id"))
    edges_for_grouping = intra_partition_edges.select("partition_id", F.lit("edge").alias("type"), F.lit(None).alias("global_node_id"), F.lit(None).cast(ArrayType(FloatType())).alias("features"), F.lit(None).cast(IntegerType()).alias("label"), F.lit(None).cast(BooleanType()).alias("train_mask"), F.lit(None).cast(BooleanType()).alias("val_mask"), F.lit(None).cast(BooleanType()).alias("test_mask"), "src_global", "dst_global", "global_edge_id")
    grouped_data = nodes_for_grouping.unionByName(edges_for_grouping)
    
    b_total_num_nodes = spark.sparkContext.broadcast(total_num_nodes)
    namenode_host = config["hdfs_base_uri"].split('//')[1].split(':')[0]
    b_webhdfs_url = spark.sparkContext.broadcast(f"http://{namenode_host}:50070")
    b_output_path = spark.sparkContext.broadcast(output_path)
    b_hdfs_user = spark.sparkContext.broadcast(config.get("hdfs_user", "root"))
    result_schema = StructType([StructField("partition_id", IntegerType()), StructField("success", BooleanType())])

    @pandas_udf(result_schema, PandasUDFType.GROUPED_MAP)
    def process_and_save_partition_fn(pdf: pd.DataFrame) -> pd.DataFrame:
        if pdf.empty: return pd.DataFrame({'partition_id': [], 'success': []})
        partition_id = int(pdf['partition_id'].iloc[0])
        success = False # Initialize success as False
        try:
            client = InsecureClient(b_webhdfs_url.value, user=b_hdfs_user.value)
            with tempfile.TemporaryDirectory() as tmpdir:
                nodes_pd = pdf[pdf['type'] == 'node'].sort_values('global_node_id').reset_index()

                # Get the expected feature dimension from the broadcast variable
                expected_feature_dim = 600

                processed_features = []
                for features_list_or_none in nodes_pd['features']:
                    if features_list_or_none is None:
                        # Handle None features: replace with a zero vector
                        processed_features.append(np.zeros(expected_feature_dim, dtype=np.float32))
                    elif not isinstance(features_list_or_none, list):
                        # Handle malformed data (e.g., scalar instead of list): replace with zero vector
                        print(f"WARNING (Partition {partition_id}): Node with unexpected feature type ({type(features_list_or_none)}). Replacing with zero vector.")
                        processed_features.append(np.zeros(expected_feature_dim, dtype=np.float32))
                    else:
                        feature_vec = np.array(features_list_or_none, dtype=np.float32)
                        
                        if feature_vec.size < expected_feature_dim:
                            # Pad with zeros if the vector is shorter
                            padded_vec = np.pad(feature_vec, (0, expected_feature_dim - feature_vec.size), 'constant')
                            processed_features.append(padded_vec)
                        elif feature_vec.size > expected_feature_dim:
                            # Truncate if the vector is longer
                            truncated_vec = feature_vec[:expected_feature_dim]
                            processed_features.append(truncated_vec)
                        else:
                            # Vector is already the correct size
                            processed_features.append(feature_vec)

                # Use vstack to create the 2D array from list of 1D arrays
                # Handle case where processed_features might be empty if nodes_pd was empty
                feature_array = np.vstack(processed_features) if processed_features else np.empty((0, expected_feature_dim), dtype=np.float32)

                feats_dict = {
                    'x': torch.from_numpy(feature_array), 
                    'y': torch.from_numpy(nodes_pd['label'].to_numpy(dtype=np.int64)), 
                    'train_mask': torch.from_numpy(nodes_pd['train_mask'].to_numpy(dtype=np.bool_)), 
                    'val_mask': torch.from_numpy(nodes_pd['val_mask'].to_numpy(dtype=np.bool_)), 
                    'test_mask': torch.from_numpy(nodes_pd['test_mask'].to_numpy(dtype=np.bool_))
                }
                node_feats = {'global_id': torch.tensor(nodes_pd['global_node_id'].values, dtype=torch.long), 'feats': feats_dict}
                node_feats_path = os.path.join(tmpdir, "node_feats.pt")
                torch.save(node_feats, node_feats_path)
                
                # Process edges (unchanged)
                edges_pd = pdf[pdf['type'] == 'edge']
                graph = {'row': torch.tensor(edges_pd['src_global'].values, dtype=torch.long), 'col': torch.tensor(edges_pd['dst_global'].values, dtype=torch.long), 'edge_id': torch.tensor(edges_pd['global_edge_id'].values, dtype=torch.long), 'size': (b_total_num_nodes.value, b_total_num_nodes.value)} if not edges_pd.empty else {'row': torch.empty(0, dtype=torch.long), 'col': torch.empty(0, dtype=torch.long), 'edge_id': torch.empty(0, dtype=torch.long), 'size': (b_total_num_nodes.value, b_total_num_nodes.value)}
                graph_path, edge_feats_path = os.path.join(tmpdir, "graph.pt"), os.path.join(tmpdir, "edge_feats.pt")
                torch.save(graph, graph_path); torch.save(defaultdict(), edge_feats_path)
                
                # Upload files to HDFS
                hdfs_partition_dir = f"{b_output_path.value}/part_{partition_id}"
                client.upload(f"{hdfs_partition_dir}/node_feats.pt", node_feats_path, overwrite=True)
                client.upload(f"{hdfs_partition_dir}/graph.pt", graph_path, overwrite=True)
                client.upload(f"{hdfs_partition_dir}/edge_feats.pt", edge_feats_path, overwrite=True)
            success = True
        except Exception as e:
            print(f"--- START: FULL STACK TRACE FOR ERROR IN PARTITION {partition_id} ---")
            print(f"ERROR processing partition {partition_id}: {e}")
            traceback.print_exc()
            print(f"--- END: FULL STACK TRACE FOR ERROR IN PARTITION {partition_id} ---")
            success = False
        return pd.DataFrame([{'partition_id': partition_id, 'success': success}])

    result = grouped_data.groupBy("partition_id").apply(process_and_save_partition_fn)
    failed_partitions = result.where(F.col("success") == False).count()
    if failed_partitions > 0: logger.error(f"{failed_partitions} partitions failed to save. Check executor logs.")
    else: logger.info("All partitions processed and saved successfully.")

def main_1():
    """Main execution function with direct-to-driver map generation."""
    spark = None
    try:
        spark = create_spark_session(CONFIG)
        
        # --- Data Loading and Filtering ---
        nodes_subset_df, all_edges_df = load_from_nebula(spark, CONFIG)
        nodes_subset_df.cache()
        logger.info(f"COUNT: nodes_subset_df after loading and limiting: {nodes_subset_df.count()}")
        # nodes_subset_df.select("id", "cust_id").show(5, False) # Show sample cust_ids

        logger.info("Filtering edges for graph consistency...")
        node_ids = nodes_subset_df.select("id").distinct()
        edges_subset_df = all_edges_df \
            .join(node_ids.withColumnRenamed("id", "src"), "src", "inner") \
            .join(node_ids.withColumnRenamed("id", "dst"), "dst", "inner")
        logger.info(f"COUNT: edges_subset_df after filtering: {edges_subset_df.count()}")

        # --- Scalable ID Generation ---
        logger.info("Generating global 0-indexed IDs...")
        node_id_map_rdd = nodes_subset_df.select("id", "cust_id").distinct().rdd.map(lambda r: (r.id, r.cust_id)).zipWithIndex()
        node_id_map = node_id_map_rdd.map(lambda x: (x[0][0], x[0][1], x[1])).toDF(["id", "cust_id", "global_node_id"])
        node_id_map.cache()
        logger.info(f"COUNT: node_id_map (all 3M nodes): {node_id_map.count()}")
        # node_id_map.show(5, False) # Show sample node_id_map entries

        src_map = node_id_map.select("id", "global_node_id").withColumnRenamed("id", "src").withColumnRenamed("global_node_id", "src_global")
        dst_map = node_id_map.select("id", "global_node_id").withColumnRenamed("id", "dst").withColumnRenamed("global_node_id", "dst_global")
        edges_w_global_ids = edges_subset_df.join(src_map, "src", "inner").join(dst_map, "dst", "inner")
        
        edge_id_rdd = edges_w_global_ids.select("src_global", "dst_global").rdd.zipWithIndex()
        edge_id_df = edge_id_rdd.map(lambda x: (x[0][0], x[0][1], x[1])).toDF(["src_global", "dst_global", "global_edge_id"])
        edges_df = edges_w_global_ids.join(edge_id_df, ["src_global", "dst_global"], "inner")
        
        edges_df.cache()
        total_num_nodes = node_id_map.count()
        total_num_edges = edges_df.count()
        logger.info(f"COUNT: edges_df after ID generation: {edges_df.count()}")
        logger.info(f"Subset Total nodes (from node_id_map): {total_num_nodes}, Subset Total edges: {total_num_edges}")

        # --- Graph Partitioning and Feature Enrichment ---
        partitioned_nodes_df = partition_graph(nodes_subset_df, edges_subset_df, CONFIG["partitions"])
        logger.info(f"COUNT: partitioned_nodes_df after graph partitioning: {partitioned_nodes_df.count()}")
        # partitioned_nodes_df.show(5, False)

        # --- CRITICAL JOIN POINT 1 ---
        # This join links node IDs from partitioning to global IDs from node_id_map
        partitioned_nodes_with_global_id = partitioned_nodes_df.join(node_id_map, "id", "inner")
        logger.info(f"COUNT: partitioned_nodes_with_global_id after join with node_id_map: {partitioned_nodes_with_global_id.count()}")
        # partitioned_nodes_with_global_id.select("id", "cust_id", "partition_id", "global_node_id").show(5, False)
        
        nodes_subset_df.unpersist()
        
        cassandra_features_df = spark.read.format("org.apache.spark.sql.cassandra") \
        .options(table=CONFIG['cassandra']['table'], keyspace=CONFIG['cassandra']['keyspace']) \
        .load()

        missing_nodes_df = partitioned_nodes_with_global_id.join(
            cassandra_features_df,
            "cust_id",
            "anti" # The "anti" join type is key here
        )

        logger.info(f"COUNT: Found {missing_nodes_df.count()} nodes that are in Nebula but missing from Cassandra.")
        logger.info("COUNT: Sample of missing cust_ids:")
        missing_nodes_df.select("cust_id").show(20, False)

        # --- CRITICAL JOIN POINT 2 ---
        # This join links node IDs to Cassandra features
        enriched_nodes_df = enrich_data_from_cassandra(spark, partitioned_nodes_with_global_id, CONFIG)
        logger.info(f"COUNT: enriched_nodes_df after Cassandra enrichment: {enriched_nodes_df.count()}")
        # enriched_nodes_df.select("id", "cust_id", "partition_id", "global_node_id", "features").show(5, False)
        
        final_nodes_df = add_temporal_train_test_split(enriched_nodes_df)
        final_nodes_df.cache()
        logger.info(f"COUNT: final_nodes_df after temporal split: {final_nodes_df.count()}")

        # --- Distributed Data Saving ---
        logger.info("Starting distributed save of partitioned data files...")
        save_partitioned_data(spark, final_nodes_df, edges_df, total_num_nodes, CONFIG)
        logger.info("Distributed data saving complete.")

        # --- Centralized Map Generation on Driver ---
        logger.info("Starting generation of global map files on the driver...")
        generate_and_save_maps_on_driver(final_nodes_df, edges_df, total_num_nodes, total_num_edges, CONFIG)
        logger.info("Global map file generation complete.")

        # --- Cleanup ---
        edges_df.unpersist()
        final_nodes_df.unpersist()
        node_id_map.unpersist()

        logger.info("GNN data preparation pipeline finished successfully.")

    except Exception as e:
        logger.error(f"An error occurred during the Spark job: {e}", exc_info=True)
        traceback.print_exc()
        sys.exit(1)
    finally:
        if spark:
            logger.info("Stopping Spark session.")
            spark.stop()

def main_2():
    """Main execution function with fixes for edge explosion."""
    spark = None
    try:
        spark = create_spark_session(CONFIG)
        
        # --- Data Loading and Filtering ---
        nodes_subset_df, all_edges_df = load_from_nebula(spark, CONFIG)
        nodes_subset_df.cache()
        logger.info(f"COUNT: Initial nodes loaded: {nodes_subset_df.count()}")

        logger.info("Filtering edges for graph consistency...")
        node_ids = nodes_subset_df.select("id").distinct()
        edges_subset_df = all_edges_df \
            .join(node_ids.withColumnRenamed("id", "src"), "src", "inner") \
            .join(node_ids.withColumnRenamed("id", "dst"), "dst", "inner")
        
        # --- Scalable ID Generation (with Edge Explosion Fix) ---
        logger.info("Generating global 0-indexed IDs...")
        node_id_map_rdd = nodes_subset_df.select("id", "cust_id").distinct().rdd.map(lambda r: (r.id, r.cust_id)).zipWithIndex()
        node_id_map = node_id_map_rdd.map(lambda x: (x[0][0], x[0][1], x[1])).toDF(["id", "cust_id", "global_node_id"])
        node_id_map.cache()

        src_map = node_id_map.select("id", "global_node_id").withColumnRenamed("id", "src").withColumnRenamed("global_node_id", "src_global")
        dst_map = node_id_map.select("id", "global_node_id").withColumnRenamed("id", "dst").withColumnRenamed("global_node_id", "dst_global")
        edges_w_global_ids = edges_subset_df.join(src_map, "src", "inner").join(dst_map, "dst", "inner")

        # ******************** BUG FIX IS HERE ********************
        # De-duplicate the edges based on source and destination before assigning unique IDs.
        # This prevents the join explosion.
        distinct_edges = edges_w_global_ids.select("src_global", "dst_global").distinct()
        logger.info(f"COUNT: Number of unique edges to process: {distinct_edges.count()}")
        
        edge_id_rdd = distinct_edges.rdd.zipWithIndex()
        edge_id_df = edge_id_rdd.map(lambda x: (x[0][0], x[0][1], x[1])).toDF(["src_global", "dst_global", "global_edge_id"])
        
        # Join back to the original (potentially with duplicates) if you need to retain edge features later,
        # or join to the distinct set if you want a simple graph. We join to the distinct set.
        edges_df = edge_id_df
        # **********************************************************
        
        edges_df.cache()
        total_num_nodes = node_id_map.count()
        total_num_edges = edges_df.count()
        logger.info(f"COUNT (Corrected): Total unique edges: {total_num_edges}")

        # --- Graph Partitioning and Feature Enrichment ---
        partitioned_nodes_df = partition_graph(nodes_subset_df, edges_subset_df, CONFIG["partitions"])
        partitioned_nodes_with_global_id = partitioned_nodes_df.join(node_id_map, "id", "inner")
        nodes_subset_df.unpersist()

        enriched_nodes_df = enrich_data_from_cassandra(spark, partitioned_nodes_with_global_id, CONFIG)
        enriched_nodes_df.cache() # Cache immediately to stabilize the count
        logger.info(f"COUNT: Nodes after Cassandra enrichment: {enriched_nodes_df.count()}")

        final_nodes_df = add_temporal_train_test_split(enriched_nodes_df)
        final_nodes_df.cache()

        # --- Distributed Data Saving & Centralized Map Generation ---
        logger.info("Starting distributed save of partitioned data files...")
        save_partitioned_data(spark, final_nodes_df, edges_df, total_num_nodes, CONFIG)
        
        logger.info("Starting generation of global map files on the driver...")
        generate_and_save_maps_on_driver(final_nodes_df, edges_df, total_num_nodes, total_num_edges, CONFIG)
        
        logger.info("GNN data preparation pipeline finished successfully.")

    except Exception as e:
        logger.error(f"An error occurred during the Spark job: {e}", exc_info=True)
        traceback.print_exc()
        sys.exit(1)
    finally:
        if spark:
            logger.info("Stopping Spark session.")
            spark.stop()

def main_3():
    """
    Main execution function with fixes for non-determinism by adding caching
    at critical stages.
    """
    spark = None
    try:
        spark = create_spark_session(CONFIG)
        
        # --- Data Loading and Filtering ---
        nodes_subset_df = load_from_nebula(spark, CONFIG)[0]
        
        # Cache the DataFrame immediately after the non-deterministic 'limit' operation.
        # Every subsequent action in this job will now use this exact same set of 3M nodes.
        nodes_subset_df.cache()
        logger.info(f"COUNT: Initial nodes loaded and stabilized: {nodes_subset_df.count()}")

        logger.info("Filtering edges for graph consistency...")
        node_ids = nodes_subset_df.select("id").distinct()
        edges_subset_df = load_from_nebula(spark, CONFIG)[1] \
            .join(node_ids.withColumnRenamed("id", "src"), "src", "inner") \
            .join(node_ids.withColumnRenamed("id", "dst"), "dst", "inner")

        # --- Scalable ID Generation ---
        logger.info("Generating global 0-indexed IDs...")
        node_id_map_rdd = nodes_subset_df.select("id", "cust_id").distinct().rdd.map(lambda r: (r.id, r.cust_id)).zipWithIndex()
        node_id_map = node_id_map_rdd.map(lambda x: (x[0][0], x[0][1], x[1])).toDF(["id", "cust_id", "global_node_id"])
        node_id_map.cache()

        src_map = node_id_map.select("id", "global_node_id").withColumnRenamed("id", "src").withColumnRenamed("global_node_id", "src_global")
        dst_map = node_id_map.select("id", "global_node_id").withColumnRenamed("id", "dst").withColumnRenamed("global_node_id", "dst_global")
        edges_w_global_ids = edges_subset_df.join(src_map, "src", "inner").join(dst_map, "dst", "inner")

        distinct_edges = edges_w_global_ids.select("src_global", "dst_global").distinct()
        
        # Cache the unique edges before performing any actions on them. This prevents
        # the re-computation that caused the count to double.
        distinct_edges.cache()
        logger.info(f"COUNT: Number of unique edges to process: {distinct_edges.count()}")
        
        edge_id_rdd = distinct_edges.rdd.zipWithIndex()
        edge_id_df = edge_id_rdd.map(lambda x: (x[0][0], x[0][1], x[1])).toDF(["src_global", "dst_global", "global_edge_id"])
        edges_df = edge_id_df
        
        edges_df.cache()
        total_num_nodes = node_id_map.count()
        total_num_edges = edges_df.count() # This count will now be correct and match the one above.
        logger.info(f"COUNT (Corrected): Total nodes: {total_num_nodes}")
        logger.info(f"COUNT (Corrected): Total unique edges: {total_num_edges}")

        # --- Graph Partitioning and Feature Enrichment ---
        partitioned_nodes_df = partition_graph(nodes_subset_df, edges_subset_df, CONFIG["partitions"])
        partitioned_nodes_with_global_id = partitioned_nodes_df.join(node_id_map, "id", "inner")
        nodes_subset_df.unpersist()
        
        #*************************************************************************************************************
        # Load the Cassandra DataFrame once for the check
        #cassandra_features_df = spark.read.format("org.apache.spark.sql.cassandra") \
        #    .options(table=CONFIG['cassandra']['table'], keyspace=CONFIG['cassandra']['keyspace']) \
        #    .load()

        # Use the full, supported join type "left_anti" instead of "anti"
        #missing_nodes_df_1 = partitioned_nodes_with_global_id.join(
        #    cassandra_features_df,
        #    "cust_id",
        #    "left_anti" 
        #)

        # Materialize the result to get an accurate count
        #missing_nodes_df_1.cache()
        #missing_count_1 = missing_nodes_df_1.count()

        #logger.info(f"DEBUG 1: Found {missing_count_1} nodes that are in Nebula but are missing features in Cassandra.")

        #*************************************************************************************************************
        # Use the full, supported join type "left_anti" instead of "anti"
        #missing_nodes_df_2 = cassandra_features_df.join(
        #    partitioned_nodes_with_global_id,
        #    "cust_id",
        #    "left_anti" 
        #)

        # Materialize the result to get an accurate count
        #missing_nodes_df_2.cache()
        #missing_count_2 = missing_nodes_df_2.count()

        #logger.info(f"DEBUG 2: Found {missing_count_2} nodes that are in Nebula but are missing features in Nebula.")
        
        #*************************************************************************************************************
        enriched_nodes_df = enrich_data_from_cassandra(spark, partitioned_nodes_with_global_id, CONFIG)
        enriched_nodes_df.cache()
        logger.info(f"COUNT: Nodes after Cassandra enrichment: {enriched_nodes_df.count()}")

        final_nodes_df = add_temporal_train_test_split(enriched_nodes_df)
        final_nodes_df.cache()

        # --- Final Saving Steps ---
        logger.info("Starting distributed save of partitioned data files...")
        save_partitioned_data(spark, final_nodes_df, edges_df, total_num_nodes, CONFIG)
        
        logger.info("Starting generation of global map files on the driver...")
        generate_and_save_maps_on_driver(final_nodes_df, edges_df, total_num_nodes, total_num_edges, CONFIG)
        
        logger.info("GNN data preparation pipeline finished successfully.")

    except Exception as e:
        logger.error(f"An error occurred during the Spark job: {e}", exc_info=True)
        traceback.print_exc()
        sys.exit(1)
    finally:
        if spark:
            logger.info("Stopping Spark session.")
            spark.stop()

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
        # This join immediately filters our nodes down to the final set we will use for training.
        enriched_nodes_df = enrich_data_from_cassandra(spark, initial_nodes_df, CONFIG)
        enriched_nodes_df.cache()
        
        final_node_count = enriched_nodes_df.count()
        logger.info(f"COUNT: Final number of valid nodes with features: {final_node_count}")

        # --- Stage 2: Filter Edges and Generate Final, Dense IDs ---
        logger.info("Filtering edge list to only include edges between valid nodes...")
        valid_node_ids = enriched_nodes_df.select("id").distinct()
        
        # An edge is kept only if both its source and destination are in our valid node set.
        final_edges_subset_df = all_edges_df \
            .join(valid_node_ids.withColumnRenamed("id", "src"), "src", "inner") \
            .join(valid_node_ids.withColumnRenamed("id", "dst"), "dst", "inner")

        logger.info("Generating final, dense, 0-based IDs for nodes and edges...")
        
        # 1. Generate dense node IDs for our final set of nodes.
        # The 'global_node_id' created here is now final and sequential from 0 to N-1.
        node_id_map_rdd = enriched_nodes_df.select("id", "cust_id").distinct().rdd.map(lambda r: (r.id, r.cust_id)).zipWithIndex()
        final_node_id_map = node_id_map_rdd.map(lambda x: (x[0][0], x[0][1], x[1])).toDF(["id", "cust_id", "global_node_id"])
        final_node_id_map.cache()

        # 2. Use the new dense node IDs to create the final edge list.
        src_map = final_node_id_map.select("id", "global_node_id").withColumnRenamed("id", "src").withColumnRenamed("global_node_id", "src_global")
        dst_map = final_node_id_map.select("id", "global_node_id").withColumnRenamed("id", "dst").withColumnRenamed("global_node_id", "dst_global")
        edges_w_dense_node_ids = final_edges_subset_df.join(src_map, "src", "inner").join(dst_map, "dst", "inner")

        # 3. Generate dense edge IDs for our final set of edges.
        distinct_edges = edges_w_dense_node_ids.select("src_global", "dst_global").distinct()
        edge_id_rdd = distinct_edges.rdd.zipWithIndex()
        final_edges_df = edge_id_rdd.map(lambda x: (x[0][0], x[0][1], x[1])).toDF(["src_global", "dst_global", "global_edge_id"])
        final_edges_df.cache()

        total_num_nodes = final_node_count # We already have this count
        total_num_edges = final_edges_df.count()
        logger.info(f"COUNT (Final): Total nodes for training: {total_num_nodes}")
        logger.info(f"COUNT (Final): Total unique edges for training: {total_num_edges}")

        # --- Stage 3: Partitioning and Final Data Preparation ---
        # Join all the pieces together into the final DataFrame for saving.
        nodes_with_ids = enriched_nodes_df.join(final_node_id_map, ["id", "cust_id"], "inner")

        logger.info("Partitioning the final, valid graph...")
        # Note: We pass the subset of edges to the partitioning algorithm.
        partitioned_nodes_df = partition_graph(nodes_with_ids, final_edges_subset_df, CONFIG["partitions"])
        
        nodes_with_partitions = nodes_with_ids.join(partitioned_nodes_df, "id", "inner")
        
        # Add temporal splits to the final node set.
        final_nodes_df = add_temporal_train_test_split(nodes_with_partitions)
        final_nodes_df.cache()

        # --- Stage 4: Saving ---
        logger.info("Starting distributed save of partitioned data files...")
        save_partitioned_data(spark, final_nodes_df, final_edges_df, total_num_nodes, CONFIG)
        
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
