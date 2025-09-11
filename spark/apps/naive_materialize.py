from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# --- Configuration ---
HDFS_INPUT_PATH = "hdfs://namenode:8020/telecom/customer_features"
CASSANDRA_HOST = "cassandra"  
CASSANDRA_PORT = "9042"
CASSANDRA_KEYSPACE = "feature_store"
CASSANDRA_TABLE = "customer_features"
CASSANDRA_DATACENTER = "datacenter1"
SPARK_PACKAGES = (
    "com.datastax.spark:spark-cassandra-connector_2.11:2.4.3,"
    "com.twitter:jsr166e:1.1.0"
)

# --- Main Spark Application ---

if __name__ == "__main__":
    print("Starting Spark job to load Parquet data from HDFS to Cassandra.")

    spark = (
        SparkSession.builder.appName("HdfsToCassandraLoader")
        .config("spark.jars.packages", SPARK_PACKAGES)
        .config("spark.cassandra.connection.host", CASSANDRA_HOST)
        .config("spark.cassandra.connection.port", CASSANDRA_PORT)
        .config("spark.cassandra.connection.local_dc", CASSANDRA_DATACENTER)
        .getOrCreate()
    )

    # --- Step 1: Read Parquet data from HDFS ---
    try:
        print(f"Reading Parquet files from: {HDFS_INPUT_PATH}")
        features_df = spark.read.parquet(HDFS_INPUT_PATH)

        print("Schema of the loaded DataFrame:")
        features_df.printSchema()
        print("Sample data from HDFS:")
        features_df.show(5, truncate=False)

    except Exception as e:
        print(f"Error reading data from HDFS. Please check the path and HDFS status.")
        print(f"Error details: {e}")
        spark.stop()
        exit(1)


    # --- Step 2: Write DataFrame to Cassandra ---
    try:
        print(f"Writing data to Cassandra table: {CASSANDRA_KEYSPACE}.{CASSANDRA_TABLE}")
        
        # Ensure DataFrame column names match Cassandra table column names
        # The connector automatically maps them by name.
        # Our current script `customer_feature_generation.py` already produces matching names.
        
        (
            features_df.write
            .format("org.apache.spark.sql.cassandra")  # Specify the Cassandra data source
            .options(table=CASSANDRA_TABLE, keyspace=CASSANDRA_KEYSPACE)
            .mode("append")  # Use "append" to add new data. Use "overwrite" to replace all data.
            .save()
        )
        
        print("Successfully wrote data to Cassandra.")

    except Exception as e:
        print(f"Error writing data to Cassandra. Please check Cassandra's status and the table schema.")
        print(f"Error details: {e}")
        spark.stop()
        exit(1)


    # --- Job Completion ---
    print("Spark job finished successfully.")
    spark.stop()
