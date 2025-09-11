import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# --- Configuration ---
HDFS_INPUT_PATH = "hdfs://namenode:8020/telecom/customer_features"
CASSANDRA_HOST = "cassandra"
CASSANDRA_PORT = "9042"
CASSANDRA_KEYSPACE = "feature_store"
CASSANDRA_TABLE = "customer_features"

# The timestamp column in your Parquet files used for filtering.
# This must match the actual column name in your data.
TIMESTAMP_COLUMN = "timestamp" 

# Add the required dependency for the `LongAdder` class error.
SPARK_PACKAGES = (
    "com.datastax.spark:spark-cassandra-connector_2.11:2.4.3,"
    "com.twitter:jsr166e:1.1.0"
)

# --- Main Spark Application ---

if __name__ == "__main__":
    # --- Step 1: Parameterization - Read Command-Line Arguments ---
    if len(sys.argv) != 3:
        print("Usage: spark-submit <script_name>.py <start_timestamp> <end_timestamp>", file=sys.stderr)
        print("Example: spark-submit incremental_load_to_cassandra.py '2025-09-09 00:00:00' '2025-09-10 00:00:00'", file=sys.stderr)
        sys.exit(1)

    start_timestamp = sys.argv[1]
    end_timestamp = sys.argv[2]

    print(f"Starting incremental load for time range: {start_timestamp} to {end_timestamp}")

    spark = (
        SparkSession.builder.appName("IncrementalHdfsToCassandra")
        .config("spark.jars.packages", SPARK_PACKAGES)
        .config("spark.cassandra.connection.host", CASSANDRA_HOST)
        .config("spark.cassandra.connection.port", CASSANDRA_PORT)
        .getOrCreate()
    )

    # --- Step 2: Incremental Filtering - Read and Filter HDFS Data ---
    try:
        print(f"Reading Parquet files from: {HDFS_INPUT_PATH}")
        
        # Read the entire dataset and then apply a filter.
        # Spark's Parquet reader supports "predicate pushdown," which efficiently
        # skips reading irrelevant data files or row groups based on the filter condition.
        incremental_df = (
            spark.read.parquet(HDFS_INPUT_PATH)
            .where(
                (col(TIMESTAMP_COLUMN) >= start_timestamp) & 
                (col(TIMESTAMP_COLUMN) < end_timestamp)
            )
        )

        # Cache the DataFrame if you need to perform multiple actions on it.
        # This is useful to avoid re-reading and re-filtering the data from HDFS.
        incremental_df.cache()

        record_count = incremental_df.count()
        if record_count == 0:
            print("No new or updated records found in the specified time range. Exiting.")
            spark.stop()
            sys.exit(0)

        print(f"Found {record_count} records to load/update.")
        print("Schema of the filtered DataFrame:")
        incremental_df.printSchema()
        print("Sample of records to be upserted:")
        incremental_df.show(5, truncate=False)

    except Exception as e:
        print(f"Error reading or filtering data from HDFS.")
        print(f"Error details: {e}")
        spark.stop()
        sys.exit(1)

    # --- Step 3: Upsert to Cassandra ---
    try:
        print(f"Writing {record_count} records to Cassandra table: {CASSANDRA_KEYSPACE}.{CASSANDRA_TABLE}")
        
        # Cassandra handles writes as "upserts" by default.
        # - If a row with the provided primary key (cust_id) exists, it will be updated.
        # - If it does not exist, a new row will be inserted.
        # The `append` mode is the correct choice for this operation.
        (
            incremental_df.write
            .format("org.apache.spark.sql.cassandra")
            .options(table=CASSANDRA_TABLE, keyspace=CASSANDRA_KEYSPACE)
            .mode("append") 
            .save()
        )
        
        print("Successfully wrote data to Cassandra.")

    except Exception as e:
        print(f"Error writing data to Cassandra.")
        print(f"Error details: {e}")
        spark.stop()
        sys.exit(1)
    finally:
        # Unpersist the cached DataFrame to free up memory
        incremental_df.unpersist()


    # --- Job Completion ---
    print("Spark incremental load job finished successfully.")
    spark.stop()
