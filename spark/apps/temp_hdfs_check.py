import os
from hdfs import InsecureClient
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Match the HDFS configuration from your main Spark application
HDFS_BASE_URI = "hdfs://namenode:8020"
WEBHDFS_URL = "http://namenode:50070"  # WebHDFS is typically on port 50070 or 9870
HDFS_USER = "root"

# Directories to test, matching your Spark app's temporary paths
DIRS_TO_TEST = [
    "/tmp/pyg_dataset/_tmp_maps/node_map_parts",
    "/tmp/pyg_dataset/_tmp_maps/edge_map_parts",
]

def test_hdfs_write_permission():
    """
    Connects to HDFS and attempts to write a small test file to the specified directories.
    """
    logging.info(f"Connecting to WebHDFS at {WEBHDFS_URL} as user '{HDFS_USER}'")
    try:
        client = InsecureClient(WEBHDFS_URL, user=HDFS_USER)
    except Exception as e:
        logging.error(f"Failed to connect to HDFS. Please check your HDFS URI and that WebHDFS is enabled. Error: {e}")
        return

    for hdfs_dir in DIRS_TO_TEST:
        test_file_path = os.path.join(hdfs_dir, "_test_write_permission.tmp")
        logging.info(f"--- Testing directory: {hdfs_dir} ---")
        
        try:
            # 1. Ensure the parent directory exists for the test file
            # The `makedirs` function is idempotent and won't fail if the directory already exists.
            logging.info(f"Attempting to create directory structure: {hdfs_dir}")
            client.makedirs(hdfs_dir)
            logging.info("Directory structure confirmed.")

            # 2. Attempt to write a small file
            logging.info(f"Attempting to write a test file to: {test_file_path}")
            with client.write(test_file_path, overwrite=True) as writer:
                writer.write(b"permission test")
            
            logging.info(f"SUCCESS: Successfully wrote to {test_file_path}")

        except Exception as e:
            logging.error(f"FAILED: Could not write to {hdfs_dir}. Error: {e}")
            logging.error("This is likely a HDFS permission issue. Check the owner and permissions of the parent directories (e.g., /tmp/pyg_dataset).")
            
        finally:
            # 3. Clean up the test file if it exists
            try:
                status = client.status(test_file_path, strict=False)
                if status:
                    client.delete(test_file_path)
                    logging.info(f"Cleaned up test file: {test_file_path}")
            except Exception as e:
                # This might fail if the write failed, which is okay.
                logging.warning(f"Could not clean up test file, which might be expected if the write failed. Warning: {e}")
        print("-" * 20)


if __name__ == "__main__":
    test_hdfs_write_permission()
