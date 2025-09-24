import os
from pyspark.sql import SparkSession

# --- Configuration ---
HDFS_BASE_URI = "hdfs://namenode:8020"
DIRS_TO_TEST = [
    "/tmp/pyg_dataset/_tmp_maps/node_map_parts",
    "/tmp/pyg_dataset/_tmp_maps/edge_map_parts",
]

def main():
    spark = SparkSession.builder.appName("HDFS_Permission_Test").getOrCreate()
    sc = spark.sparkContext

    print("--- Starting HDFS Write Permission Test from Spark ---")

    # Create a tiny DataFrame to write
    test_df = spark.createDataFrame([("test", 1)], ["data", "id"])

    for hdfs_dir in DIRS_TO_TEST:
        # We write to a subdirectory to avoid conflicting with potential existing data
        test_output_path = os.path.join(HDFS_BASE_URI, hdfs_dir, "spark_permission_test")
        print(f"\n--- Testing write to: {test_output_path} ---")

        try:
            test_df.write.mode("overwrite").parquet(test_output_path)
            print(f"SUCCESS: Spark job successfully wrote to {test_output_path}")

            # Simple cleanup using Spark's Hadoop FileSystem API
            print("Cleaning up test directory...")
            fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(sc._jsc.hadoopConfiguration())
            path = sc._jvm.org.apache.hadoop.fs.Path(test_output_path)
            if fs.exists(path):
                fs.delete(path, True) # True for recursive delete
            print("Cleanup successful.")

        except Exception as e:
            print(f"FAILED: Spark job could not write to {test_output_path}.")
            print("This confirms a permission issue for the Spark executor user.")
            print(f"Full Error: {e}")
            break # Stop on first failure
    
    print("\n--- HDFS Write Permission Test Finished ---")
    spark.stop()

if __name__ == "__main__":
    main()
