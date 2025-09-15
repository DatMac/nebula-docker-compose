# test_pipeline.py (Corrected for Spark 2.4 apply API)

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

# --- THIS IS A NEW, REQUIRED IMPORT FOR THE SPARK 2.4 API ---
from pyspark.sql.functions import pandas_udf, PandasUDFType


# --- Paste your original CONFIG, create_spark_session, and enrich_data_from_cassandra
# --- functions here. I have included them below for completeness.

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration Section ---
CONFIG = {
    "app_name": "GNN_Data_Prep_Pipeline_TEST",
    "partitions": 2,
    "spark_master_url": "spark://spark-master:7077",
    "spark_packages": [
        "com.vesoft.nebula-spark-connector:nebula-spark-connector_2.2:3.4.0",
        "com.datastax.spark:spark-cassandra-connector_2.11:2.4.3",
        "graphframes:graphframes:0.8.1-spark2.4-s_2.11",
        "com.twitter:jsr166e:1.1.0"
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

def create_spark_session(config: Dict[str, Any]) -> SparkSession:
    """Initializes and returns a SparkSession."""
    logger.info("Initializing Spark session...")
    spark_builder = (
        SparkSession.builder.appName(config["app_name"])
        .master(config["spark_master_url"])
        .config("spark.jars.packages", ",".join(config["spark_packages"]))
        .config(f"spark.cassandra.connection.host", config["cassandra"]["host"])
        .config(f"spark.cassandra.connection.port", config["cassandra"]["port"])
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "10000")
    )
    spark = spark_builder.getOrCreate()
    spark.sparkContext.setCheckpointDir(config["checkpoint_dir"])
    logger.info("Spark session created successfully.")
    return spark

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
    
    count = enriched_nodes_df.count()
    logger.info(f"Successfully enriched {count} nodes from Cassandra.")
    if count == 0:
        logger.warning("WARNING: No matching nodes found in Cassandra. Ensure your sample data is inserted correctly.")
    
    return enriched_nodes_df

# ==============================================================================
#  MODIFIED SECTION: UDF Factory and Save Function
# ==============================================================================

def create_partition_processor_udf(config: Dict[str, Any]):
    """
    Factory function to create and configure the Pandas UDF for Spark 2.4.
    This pattern allows us to pass configuration into the UDF's scope.
    """
    output_path = config["output_path"]
    hdfs_base_uri = config["hdfs_base_uri"]

    # Define the schema for the UDF's output here
    result_schema = StructType([
        StructField("partition_id", IntegerType()),
        StructField("success", BooleanType())
    ])
    
    @pandas_udf(result_schema, PandasUDFType.GROUPED_MAP)
    def process_partition_group(pdf: pd.DataFrame) -> pd.DataFrame:
        if pdf.empty: return pd.DataFrame({'partition_id': [], 'success': []})
        partition_id = int(pdf['partition_id'].iloc[0])
        try:
            # --- THIS IS THE CRITICAL CODE CHANGE ---
            # Use the legacy pyarrow.hdfs.connect() API which exists in PyArrow 0.14.1
            # It automatically finds the namenode from the environment configuration.
            hdfs = pa.hdfs.connect()

            partition_dir = f"{output_path}/partition_{partition_id}"
            if not hdfs.exists(partition_dir):
                hdfs.mkdir(partition_dir)

            nodes_pd = pdf[pdf['type'] == 'node'].sort_values('local_node_idx')
            if not nodes_pd.empty:
                node_features = torch.tensor(np.stack(nodes_pd['features'].values), dtype=torch.float)
                labels = torch.tensor(nodes_pd['label'].values, dtype=torch.long)
                with hdfs.open(f"{partition_dir}/node_features.pt", "wb") as f: torch.save(node_features, f)
                with hdfs.open(f"{partition_dir}/labels.pt", "wb") as f: torch.save(labels, f)

            edges_pd = pdf[pdf['type'] == 'edge']
            if not edges_pd.empty:
                edge_index = torch.tensor([edges_pd['src_local'].values, edges_pd['dst_local'].values], dtype=torch.long)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            with hdfs.open(f"{partition_dir}/edge_index.pt", "wb") as f: torch.save(edge_index, f)
            success = True
        except Exception as e:
            # Log the full traceback for better debugging
            import traceback
            logging.error(f"Error processing partition {partition_id}: {e}\n{traceback.format_exc()}")
            success = False
        return pd.DataFrame([{'partition_id': partition_id, 'success': success}])
    return process_partition_group

def save_partitions_for_pyg(enriched_nodes_df: DataFrame, edges_df: DataFrame, config: Dict[str, Any]):
    """
    Formats and saves partitioned graph data to HDFS using the Spark 2.4
    decorator-based UDF pattern.
    """
    logger.info(f"Starting scalable save of {config['partitions']} partitions to HDFS.")

    # Data preparation logic remains the same
    window_spec = Window.partitionBy("partition_id").orderBy("id")
    nodes_with_local_idx = enriched_nodes_df.withColumn("local_node_idx", F.row_number().over(window_spec) - 1)
    nodes_with_local_idx.cache()
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
    nodes_for_grouping = nodes_with_local_idx.select(
        "partition_id", F.lit("node").alias("type"), F.col("local_node_idx"),
        F.col("features"), F.col("label"), F.lit(None).cast(LongType()).alias("src_local"),
        F.lit(None).cast(LongType()).alias("dst_local")
    )
    edges_for_grouping = intra_partition_edges.select(
        "partition_id", F.lit("edge").alias("type"), F.lit(None).cast(LongType()).alias("local_node_idx"),
        F.lit(None).cast(ArrayType(FloatType())).alias("features"), F.lit(None).cast(IntegerType()).alias("label"),
        "src_local", "dst_local"
    )
    grouped_data = nodes_for_grouping.unionByName(edges_for_grouping)

    # --- THIS IS THE FIX ---
    # 1. Create the UDF using the factory
    process_udf = create_partition_processor_udf(config)
    
    # 2. Call .apply() with ONLY the UDF object. No 'schema' argument.
    logger.info("Executing save operation on all partitions using '.apply' for Spark 2.4...")
    result = grouped_data.groupBy("partition_id").apply(process_udf)
    
    # Rest of the function is the same
    failed_partitions = result.where(F.col("success") == False).count()
    if failed_partitions > 0:
        logger.error(f"{failed_partitions} partitions failed to save.")
    else:
        logger.info("All partitions processed and saved successfully.")
    nodes_with_local_idx.unpersist()

# ==============================================================================
#  TESTING FUNCTION (Unchanged)
# ==============================================================================
def run_test():
    """
    Runs a targeted test for the enrichment and save functions using a small,
    in-memory sample of data.
    """
    spark = create_spark_session(CONFIG)

    logger.info("Step 1: Creating mock data for testing...")
    sample_ids = [(f'CUST-{i:08}',) for i in range(1, 11)]
    mock_nodes_df = spark.createDataFrame(sample_ids, ["id"])

    mock_edges_data = [
        ('CUST-00000002', 'CUST-00000004'), ('CUST-00000004', 'CUST-00000005'),
        ('CUST-00000006', 'CUST-00000007'), ('CUST-00000007', 'CUST-00000008'),
        ('CUST-00000009', 'CUST-00000010'), ('CUST-00000005', 'CUST-00000006') # Edge between partitions
    ]
    mock_edges_df = spark.createDataFrame(mock_edges_data, ["src", "dst"])
    mock_partitioned_nodes_df = mock_nodes_df.withColumn(
        "partition_id",
        F.when(F.col("id") < "CUST-00000006", 0).otherwise(1)
    )

    logger.info("Mock partitioned data created:")
    mock_partitioned_nodes_df.show()
    
    logger.info("\nStep 2: Testing enrich_data_from_cassandra...")
    enriched_nodes_df = enrich_data_from_cassandra(spark, mock_partitioned_nodes_df, CONFIG)
    enriched_nodes_df.cache()
    logger.info("Enriched data schema:")
    enriched_nodes_df.printSchema()
    enriched_nodes_df.show()

    logger.info("\nStep 3: Testing save_partitions_for_pyg...")
    save_partitions_for_pyg(enriched_nodes_df, mock_edges_df, CONFIG)

    logger.info("Test finished successfully.")
    enriched_nodes_df.unpersist()
    spark.stop()

if __name__ == "__main__":
    run_test()
