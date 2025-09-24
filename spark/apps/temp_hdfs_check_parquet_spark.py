import logging
import sys
from pyspark.sql import SparkSession

# --- Configuration ---
# All you need to change is here
SPARK_MASTER_URL = "spark://spark-master:7077"
# The full HDFS path including the hdfs:// prefix and port
HDFS_TEST_PATH = "hdfs://namenode:8020/tmp/pyg_dataset/temp"
APP_NAME = "HDFS_Write_Permission_Test"

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


def main():
    """
    A simple Spark job to test HDFS write permissions.
    """
    spark = None
    try:
        # 1. Create a Spark Session
        logger.info(f"Initializing Spark session with master at {SPARK_MASTER_URL}...")
        spark = (
            SparkSession.builder
            .appName(APP_NAME)
            .master(SPARK_MASTER_URL)
            .getOrCreate()
        )
        logger.info("Spark session created successfully.")

        # 2. Create a small, dummy DataFrame
        logger.info("Creating a small test DataFrame in memory...")
        data = [
            (1, "test_data_a"),
            (2, "test_data_b"),
            (3, "test_data_c"),
        ]
        columns = ["id", "test_payload"]
        df = spark.createDataFrame(data, columns)
        logger.info("Test DataFrame created:")
        df.show()

        # 3. Attempt to write to HDFS
        # The mode "overwrite" is used so you can run this test multiple times
        # without it failing because the directory already exists.
        logger.info(f"ATTEMPTING TO WRITE to HDFS path: {HDFS_TEST_PATH}")
        df.write.mode("overwrite").parquet(HDFS_TEST_PATH)

        # If the line above completes without an error, the write was successful.
        logger.info("==========================================================")
        logger.info(">>> SUCCESS: Spark job wrote to HDFS successfully.      <<<")
        logger.info("==========================================================")

        # 4. Verification Step: Read the data back
        logger.info("Verification: Attempting to read the data back from HDFS...")
        read_df = spark.read.parquet(HDFS_TEST_PATH)
        count = read_df.count()
        logger.info(f"Successfully read back {count} rows. Verification complete.")
        read_df.show()


    except Exception as e:
        logger.error("=====================================================================")
        logger.error(">>> FAILURE: Spark job FAILED to write to HDFS.                   <<<")
        logger.error(">>> This is almost certainly a HDFS PERMISSION or CONNECTION issue. <<<")
        logger.error("=====================================================================")
        logger.error("Full exception traceback:", exc_info=True)
        # Exit with a non-zero status code to indicate failure
        sys.exit(1)

    finally:
        # 5. Clean up the Spark Session
        if spark:
            logger.info("Stopping Spark session.")
            spark.stop()

if __name__ == "__main__":
    main()
