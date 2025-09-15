import os
import sys
import logging
from typing import Dict, Any

import pandas as pd
import torch
import numpy as np
import pyarrow as pa

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, LongType, IntegerType, FloatType, BooleanType
from graphframes import GraphFrame

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration Section ---
# All configurable parameters are centralized here for easy management.
CONFIG = {
    "app_name": "GNN_Data_Prep_Pipeline",
    "partitions": 8,  # Number of partitions for the graph (N)

    # Spark and Connector Configurations (Updated for Spark 3.x / Scala 2.12)
    "spark_master_url": "spark://spark-master:7077",
    "spark_packages": [
        "com.vesoft.nebula-spark-connector:nebula-spark-connector_2.2:3.4.0",
        "com.datastax.spark:spark-cassandra-connector_2.11:2.4.3",
        "graphframes:graphframes:0.8.1-spark2.4-s_2.11",
        "com.twitter:jsr166e:1.1.0"
    ],

    # HDFS Configuration
    "hdfs_base_uri": "hdfs://namenode:8020",
    "output_path": "/telecom/pyg_dataset",
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
    
    # logger.info(f"Successfully created plan to enrich {partitioned_nodes_df.count()} nodes.")
    logger.info("Successfully created plan to enrich nodes.")
    return enriched_nodes_df


def save_partitions_for_pyg(enriched_nodes_df: DataFrame, edges_df: DataFrame, config: Dict[str, Any]):
    """
    Formats and saves partitioned graph data to HDFS using a scalable
    `groupBy().applyInPandas()` approach to avoid memory bottlenecks.
    """
    logger.info(f"Starting scalable save of {config['partitions']} partitions to HDFS.")

    # 1. Assign local, 0-indexed node IDs within each partition. This is essential for PyG.
    window_spec = Window.partitionBy("partition_id").orderBy("id")
    nodes_with_local_idx = enriched_nodes_df.withColumn("local_node_idx", F.row_number().over(window_spec) - 1)
    nodes_with_local_idx.cache() # Cache for efficient reuse in joins

    # 2. Prepare edge data by mapping global IDs to local partition IDs and local node indices.
    node_map = nodes_with_local_idx.select("id", "partition_id", "local_node_idx")
    
    intra_partition_edges = (
        edges_df.join(node_map.alias("src_map"), F.col("src") == F.col("src_map.id"))
        .join(node_map.alias("dst_map"), F.col("dst") == F.col("dst_map.id"))
        .where(F.col("src_map.partition_id") == F.col("dst_map.partition_id"))
        .select(
            F.col("src_map.partition_id").alias("partition_id"),
            F.col("src_map.local_node_idx").alias("src_local"),
            F.col("dst_map.local_node_idx").alias("dst_local")
        )
    )

    # 3. Union nodes and edges into a single DataFrame with a common schema
    #    so they can be processed together in `applyInPandas`.
    nodes_for_grouping = nodes_with_local_idx.select(
        "partition_id",
        F.lit("node").alias("type"),
        F.col("local_node_idx"),
        F.col("features"),
        F.col("label"),
        F.lit(None).cast(LongType()).alias("src_local"),
        F.lit(None).cast(LongType()).alias("dst_local")
    )

    edges_for_grouping = intra_partition_edges.select(
        "partition_id",
        F.lit("edge").alias("type"),
        F.lit(None).cast(LongType()).alias("local_node_idx"),
        F.lit(None).cast(ArrayType(FloatType())).alias("features"),
        F.lit(None).cast(IntegerType()).alias("label"),
        "src_local",
        "dst_local"
    )

    grouped_data = nodes_for_grouping.unionByName(edges_for_grouping)

    # 4. Define the Pandas UDF to process each partition group.
    #    This function receives a Pandas DataFrame for one `partition_id`.
    output_path = config["output_path"]
    hdfs_base_uri = config["hdfs_base_uri"]

    def process_partition_group(pdf: pd.DataFrame) -> pd.DataFrame:
        if pdf.empty:
            return pd.DataFrame({'partition_id': [], 'success': []})

        partition_id = int(pdf['partition_id'].iloc[0])
        
        try:
            hdfs = pa.fs.HadoopFileSystem.from_uri(hdfs_base_uri)
            partition_dir = f"{output_path}/partition_{partition_id}"
            hdfs.create_dir(partition_dir, recursive=True)

            # --- Process nodes ---
            nodes_pd = pdf[pdf['type'] == 'node'].sort_values('local_node_idx')
            if not nodes_pd.empty:
                node_features = torch.tensor(np.stack(nodes_pd['features'].values), dtype=torch.float)
                labels = torch.tensor(nodes_pd['label'].values, dtype=torch.long)
                
                with hdfs.open_output_stream(f"{partition_dir}/node_features.pt") as f:
                    torch.save(node_features, f)
                with hdfs.open_output_stream(f"{partition_dir}/labels.pt") as f:
                    torch.save(labels, f)

            # --- Process edges ---
            edges_pd = pdf[pdf['type'] == 'edge']
            if not edges_pd.empty:
                edge_index = torch.tensor(
                    [edges_pd['src_local'].values, edges_pd['dst_local'].values], dtype=torch.long
                )
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            
            with hdfs.open_output_stream(f"{partition_dir}/edge_index.pt") as f:
                torch.save(edge_index, f)
            
            success = True
        except Exception as e:
            logging.error(f"Error processing partition {partition_id}: {e}")
            success = False
        
        return pd.DataFrame([{'partition_id': partition_id, 'success': success}])

    # Define the schema for the output of the Pandas UDF
    result_schema = StructType([
        StructField("partition_id", IntegerType()),
        StructField("success", BooleanType())
    ])

    # 5. Execute the operation. `groupBy().applyInPandas()` is a transformation.
    #    An action (`.count()`) is needed to trigger it.
    logger.info("Executing save operation on all partitions...")
    result = grouped_data.groupBy("partition_id").applyInPandas(process_partition_group, schema=result_schema)
    
    # Trigger the computation and check for failures
    failed_partitions = result.where(F.col("success") == False).count()
    if failed_partitions > 0:
        logger.error(f"{failed_partitions} partitions failed to save.")
    else:
        logger.info("All partitions processed and saved successfully.")

    nodes_with_local_idx.unpersist()


def main():
    """Main execution function."""
    spark = None
    try:
        spark = create_spark_session(CONFIG)
        
        nodes_df, edges_df = load_from_nebula(spark, CONFIG)
        nodes_df.cache()
        edges_df.cache()
        
        partitioned_nodes_df = partition_graph(nodes_df, edges_df, CONFIG["partitions"])
        
        enriched_nodes_df = enrich_data_from_cassandra(spark, partitioned_nodes_df, CONFIG)
        enriched_nodes_df.cache()

        save_partitions_for_pyg(enriched_nodes_df, edges_df, CONFIG)
        
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
