import numpy as np
from datetime import datetime, timedelta
import random
from pyspark.sql import Row, SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    TimestampType,
    ArrayType,
    FloatType,
)

# --- Configuration ---
TOTAL_ROWS = 10_000_000
# Adjust the number of partitions based on your Spark cluster's resources.
# A good starting point is 2-4 partitions per core in your cluster.
NUM_PARTITIONS = 10
FEATURE_DIMS = 600
HDFS_OUTPUT_PATH = "hdfs://namenode:8020/telecom/customer_features"

# --- Data Generation Logic ---

def generate_partition_data(partition_index, iterator_over_data):
    """
    This function runs on each Spark worker to generate a slice of the total data.
    `mapPartitionsWithIndex` provides the partition's index automatically.
    The `iterator_over_data` can be ignored since we generate data from scratch.
    """
    rows_per_partition = TOTAL_ROWS // NUM_PARTITIONS
    start_id = partition_index * rows_per_partition

    # We'll use Xavier (Glorot) Initialization for the feature vectors.
    # Variance = 1 / number_of_inputs.
    xavier_std_dev = np.sqrt(1.0 / FEATURE_DIMS)

    # Define the one-year time window for timestamps.
    end_time = datetime.now()
    start_time = end_time - timedelta(days=365)
    time_window_seconds = (end_time - start_time).total_seconds()

    print(f"Partition {partition_index}: Generating {rows_per_partition} rows starting from CUST-{start_id + 1:08d}")

    for i in range(rows_per_partition):
        # 1. Generate Customer ID
        cust_id = f"CUST-{start_id + i + 1:08d}"

        # 2. Generate features using Xavier initialization
        features_vector = (np.random.randn(FEATURE_DIMS) * xavier_std_dev).astype(float).tolist()

        # 3. Generate a binary label
        label = random.randint(0, 1)

        # 4. Generate a random timestamp within the window
        random_seconds = random.uniform(0, time_window_seconds)
        event_timestamp = start_time + timedelta(seconds=random_seconds)
        
        yield Row(
            cust_id=cust_id,
            features=features_vector,
            label=label,
            timestamp=event_timestamp
        )

# --- Main Spark Application ---

if __name__ == "__main__":
    spark = (
        SparkSession.builder.appName("FakeCustomerFeatureGeneration")
        .getOrCreate()
    )

    # Define the schema for the DataFrame to ensure correct data types.
    schema = StructType([
        StructField("cust_id", StringType(), False),
        StructField("features", ArrayType(FloatType()), False),
        StructField("label", IntegerType(), False),
        StructField("timestamp", TimestampType(), False),
    ])

    # Create an RDD with the desired number of empty partitions.
    # The content of the RDD doesn't matter, only its number of partitions.
    initial_rdd = spark.sparkContext.parallelize(range(NUM_PARTITIONS), NUM_PARTITIONS)

    # Use `mapPartitionsWithIndex` to generate data.
    # This provides the partition index directly to our function.
    features_rdd = initial_rdd.mapPartitionsWithIndex(generate_partition_data)

    # Convert the RDD of Row objects into a Spark DataFrame.
    features_df = spark.createDataFrame(features_rdd, schema)

    # Write the DataFrame to HDFS in Parquet format.
    print(f"Writing {TOTAL_ROWS} rows to {HDFS_OUTPUT_PATH}...")
    features_df.write.mode("overwrite").parquet(HDFS_OUTPUT_PATH)
    print("Successfully wrote features to HDFS.")

    # Stop the Spark session to release resources.
    spark.stop()
