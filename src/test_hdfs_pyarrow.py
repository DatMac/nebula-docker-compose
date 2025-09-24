# src/test_hdfs_pyarrow.py

import torch
import pyarrow.fs
import os

# --- Configuration ---
# The hostname 'namenode' is resolved by Docker's internal DNS.
NAMENODE_HOST = 'namenode'
NAMENODE_PORT = 8020
HDFS_PATH = "/test-pyarrow/my_tensor.pt"

def main():
    """
    Connects to HDFS using pyarrow, writes a file, reads it back,
    and verifies the content.
    """
    print("--- Running PyArrow HDFS Connection Test ---")
    
    # 1. Connect to HDFS
    # PyArrow will automatically use the HADOOP_CONF_DIR and CLASSPATH
    # environment variables set by our entrypoint.sh.
    try:
        print(f"Connecting to HDFS at hdfs://{NAMENODE_HOST}:{NAMENODE_PORT}...")
        hdfs = pyarrow.fs.HadoopFileSystem(host=NAMENODE_HOST, port=NAMENODE_PORT)
        print("Connection successful!")
    except Exception as e:
        print(f"!!! Test FAILED: Could not connect to HDFS. Error: {e}")
        return

    # 2. Prepare data and write to HDFS
    tensor_to_save = torch.tensor([[1., -1.], [1., -1.]])
    print(f"\nOriginal Tensor:\n{tensor_to_save}")

    try:
        print(f"Writing tensor to HDFS path: '{HDFS_PATH}'...")
        # Ensure the directory exists
        hdfs.create_dir("/test-pyarrow")
        
        with hdfs.open_output_stream(HDFS_PATH) as f:
            torch.save(tensor_to_save, f)
        print("Write operation completed.")

        # Verify the file exists
        file_info = hdfs.get_file_info(HDFS_PATH)
        print(f"Verification: File exists, size: {file_info.size} bytes.")

    except Exception as e:
        print(f"!!! Test FAILED: Could not write to HDFS. Error: {e}")
        return

    # 3. Read the data back from HDFS
    try:
        print(f"\nReading tensor back from HDFS...")
        with hdfs.open_input_stream(HDFS_PATH) as f:
            loaded_tensor = torch.load(f)
        print("Read operation completed.")
        print(f"Loaded Tensor:\n{loaded_tensor}")

    except Exception as e:
        print(f"!!! Test FAILED: Could not read from HDFS. Error: {e}")
        return
    
    finally:
        # 4. Clean up the test file, even if the read fails
        if hdfs.get_file_info(HDFS_PATH).type != pyarrow.fs.FileType.NotFound:
            print("\nCleaning up test file...")
            hdfs.delete_file(HDFS_PATH)
            print("Cleanup complete.")

    # 5. Validate the data
    if torch.equal(tensor_to_save, loaded_tensor):
        print("\n--- Test PASSED: Data integrity confirmed! ---")
    else:
        print("\n!!! Test FAILED: Loaded data does not match original data! ---")

if __name__ == "__main__":
    main()
