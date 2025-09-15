# test_hdfs_spark_job.py

import os
import sys
import traceback
import pyarrow as pa
from pyspark.sql import SparkSession

def create_spark_session():
    """Initializes and returns a minimal SparkSession."""
    print("--- [Driver] Initializing Spark Session ---")
    spark_jars_path = "/opt/bitnami/spark/jars/*"
    return (
        SparkSession.builder
        .appName("HDFS_Connection_Test_Job")
        .master("spark://spark-master:7077")
        .config("spark.driver.extraClassPath", spark_jars_path)
        .config("spark.executor.extraClassPath", spark_jars_path)
        .getOrCreate()
    )

def test_hdfs_connection_on_executor(_):
    """
    This function contains the core test logic.
    It will be sent by the driver to run on each Spark executor.
    """
    # This code runs on an executor node, not the driver.
    print("--- [Executor] Starting HDFS connection test on this executor. ---")

    # Log environment details for debugging
    print(f"--- [Executor] Python executable: {sys.executable}")
    print(f"--- [Executor] PyArrow version: {pa.__version__}")
    
    # Check for the crucial CLASSPATH environment variable
    classpath = os.environ.get("CLASSPATH")
    if classpath:
        print("--- [Executor] CLASSPATH environment variable is SET.")
        # print(f"--- [Executor] CLASSPATH value: {classpath}") # Uncomment for verbose debugging
    else:
        print("--- [Executor] WARNING: CLASSPATH environment variable is NOT SET. This is a likely cause of failure.")

    try:
        print("--- [Executor] Attempting to connect to HDFS using pa.hdfs.connect()...")
        
        # This uses the legacy API for PyArrow 0.14.1
        # It should automatically pick up Hadoop configuration if CLASSPATH is set correctly.
        hdfs = pa.hdfs.connect()

        print("--- [Executor] SUCCESS: Successfully connected to HDFS!")
        
        try:
            print("--- [Executor] Attempting to list the HDFS root directory ('/')...")
            root_dir_listing = hdfs.ls('/')
            
            print("--- [Executor] SUCCESS: Successfully listed the root directory.")
            print("--- [Executor] HDFS Root Directory Contents ---")
            for item in root_dir_listing:
                print(f"--- [Executor]   {item}")
            print("--- [Executor] ----------------------------------")

        except Exception as e:
            print("--- [Executor] FAILURE: Connected to HDFS, but failed to list directory '/'")
            print("--- [Executor] This could be a permissions issue.")
            print("--- [Executor] Full Error Traceback ---")
            traceback.print_exc()
            # Re-raise the exception to make the Spark job fail, which is more visible
            raise e

    except Exception as e:
        print("--- [Executor] FAILURE: Could not connect to HDFS.")
        print("--- [Executor] This indicates the CLASSPATH or configuration is still incorrect.")
        print("--- [Executor] Full Error Traceback ---")
        traceback.print_exc()
        # Re-raise the exception to make the Spark job fail
        raise e


def main():
    """
    Main function to orchestrate the Spark job.
    """
    spark = create_spark_session()
    
    # We need to force the `test_hdfs_connection_on_executor` function to run on the executors.
    # The easiest way is to create a dummy RDD and run an action on it.
    
    # Create an RDD with 2 partitions to run the test on 2 different executors (if available).
    num_executors_to_test = 2
    dummy_rdd = spark.sparkContext.parallelize(range(num_executors_to_test), num_executors_to_test)
    
    print(f"\n--- [Driver] Submitting test function to run on {num_executors_to_test} executors... ---")
    
    # .foreachPartition() is an action that will execute our function on each partition.
    dummy_rdd.foreachPartition(test_hdfs_connection_on_executor)
    
    print("\n--- [Driver] Spark job finished. Check the executor logs in the Spark UI for results. ---")
    
    spark.stop()


if __name__ == "__main__":
    main()
