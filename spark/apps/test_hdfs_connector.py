from pyspark.sql import SparkSession

def main():
    # Initialize Spark Session
    spark = SparkSession.builder \
        .appName("HDFSConnectionTest") \
        .getOrCreate()

    # Create a simple DataFrame
    data = [("Alice", 1), ("Bob", 2), ("Charlie", 3)]
    columns = ["name", "id"]
    df = spark.createDataFrame(data, columns)

    # HDFS file path
    hdfs_path = "hdfs://namenode:8020/tmp/test_data.csv"

    try:
        # Write DataFrame to HDFS
        print(f"Writing data to HDFS at: {hdfs_path}")
        df.write.csv(hdfs_path, header=True, mode="overwrite")
        print("Write successful!")

        # Read data back from HDFS
        print(f"Reading data from HDFS at: {hdfs_path}")
        read_df = spark.read.csv(hdfs_path, header=True)
        
        print("Read successful! Data:")
        read_df.show()

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Stop the Spark session
        spark.stop()
        print("Spark session stopped.")

if __name__ == "__main__":
    main()
