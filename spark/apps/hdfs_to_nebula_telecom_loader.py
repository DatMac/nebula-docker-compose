import re
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col

# --- Configuration ---
HDFS_BASE_PATH = "hdfs://namenode:8020/tmp/telecom"
NEBULA_META_ADDRESS = "metad0:9559,metad1:9559,metad2:9559"
NEBULA_GRAPH_ADDRESS = "graphd:9669,graphd1:9669,graphd2:9669"
NEBULA_SPACE = "telecom"
NEBULA_USER = "root"
NEBULA_PASSWORD = "nebula"
WRITE_BATCH_SIZE = 1024

def clean_dataframe_columns(df: DataFrame) -> DataFrame:
    """
    Cleans the column names of a DataFrame to match Nebula Graph property names.
    Removes special characters and type hints used in initial DataFrame creation.
    Example: 'customerID:ID(Customer)' becomes 'customerID'.
    ':START_ID(Customer)' becomes 'src'
    ':END_ID(Device)' becomes 'dst'
    ':RANK' becomes 'rank'
    """
    new_columns = []
    for c in df.columns:
        new_c = re.split(r'[:(]', c)[0]
        if ":START_ID" in c:
            new_c = "src"
        elif ":END_ID" in c:
            new_c = "dst"
        elif ":RANK" in c:
            new_c = "rank"
        new_columns.append(new_c)
    return df.toDF(*new_columns)

def write_vertices_to_nebula(spark: SparkSession, tag_name: str, vertex_id_field: str):
    """
    Reads vertex data from HDFS, cleans column names, and writes to NebulaGraph.

    :param spark: The active SparkSession.
    :param tag_name: The name of the Nebula tag (e.g., 'Customer').
    :param vertex_id_field: The DataFrame column name for the vertex ID.
    """
    print(f"\n--- Loading vertices for tag: '{tag_name}' ---")
    
    # 1. Read from HDFS
    hdfs_path = f"{HDFS_BASE_PATH}/vertices/{tag_name}"
    df = spark.read.option("header", "true").csv(hdfs_path)
    
    # 2. Clean column names
    df_clean = clean_dataframe_columns(df)
    
    print(f"Read {df_clean.count()} rows from {hdfs_path}. Schema:")
    df_clean.printSchema()

    # 3. Write to NebulaGraph
    df_clean.write.format("com.vesoft.nebula.connector.NebulaDataSource") \
        .option("type", "vertex") \
        .option("operateType", "write") \
        .option("metaAddress", NEBULA_META_ADDRESS) \
        .option("graphAddress", NEBULA_GRAPH_ADDRESS) \
        .option("spaceName", NEBULA_SPACE) \
        .option("label", tag_name) \
        .option("user", NEBULA_USER) \
        .option("passwd", NEBULA_PASSWORD) \
        .option("vertexField", vertex_id_field) \
        .option("vidPolicy", "") \
        .option("batch", WRITE_BATCH_SIZE) \
        .save()
        
    print(f"Successfully wrote vertices for tag '{tag_name}'.")

def write_edges_to_nebula(spark: SparkSession, edge_name: str, src_id_field: str = "src", dst_id_field: str = "dst", rank_field: str = ""):
    """
    Reads edge data from HDFS, cleans column names, and writes to NebulaGraph.

    :param spark: The active SparkSession.
    :param edge_name: The name of the Nebula edge (e.g., 'HAS_DEVICE').
    :param src_id_field: The DataFrame column for the source vertex ID.
    :param dst_id_field: The DataFrame column for the destination vertex ID.
    :param rank_field: The DataFrame column for the edge rank (optional).
    """
    print(f"\n--- Loading edges for type: '{edge_name}' ---")
    
    # 1. Read from HDFS
    hdfs_path = f"{HDFS_BASE_PATH}/edges/{edge_name}"
    df = spark.read.option("header", "true").csv(hdfs_path)
    
    # 2. Clean column names
    df_clean = clean_dataframe_columns(df)
    
    print(f"Read {df_clean.count()} rows from {hdfs_path}. Schema:")
    df_clean.printSchema()

    # 3. Write to NebulaGraph
    writer = df_clean.write.format("com.vesoft.nebula.connector.NebulaDataSource") \
        .option("type", "edge") \
        .option("operateType", "write") \
        .option("metaAddress", NEBULA_META_ADDRESS) \
        .option("graphAddress", NEBULA_GRAPH_ADDRESS) \
        .option("spaceName", NEBULA_SPACE) \
        .option("label", edge_name) \
        .option("user", NEBULA_USER) \
        .option("passwd", NEBULA_PASSWORD) \
        .option("srcVertexField", src_id_field) \
        .option("dstVertexField", dst_id_field) \
        .option("srcPolicy", "") \
        .option("dstPolicy", "") \
        .option("writeMode", "insert") \
        .option("batch", WRITE_BATCH_SIZE)

    if rank_field:
        writer.option("rankField", rank_field)
    else:
        writer.option("rankField", "")

    writer.save()
        
    print(f"Successfully wrote edges for type '{edge_name}'.")


def main():
    # Initialize Spark Session
    spark = SparkSession.builder \
        .appName("HDFS to NebulaGraph Telecom Loader") \
        .master("spark://spark-master:7077") \
        .getOrCreate()

    print("Spark Session created. Starting data load process into NebulaGraph...")

    # Load all vertex (Tag) data
    write_vertices_to_nebula(spark, tag_name="Customer", vertex_id_field="customerID")
    write_vertices_to_nebula(spark, tag_name="Device", vertex_id_field="deviceID")
    write_vertices_to_nebula(spark, tag_name="Phone_Number", vertex_id_field="phoneNumberID")
    write_vertices_to_nebula(spark, tag_name="Cell_Tower", vertex_id_field="towerID")
    write_vertices_to_nebula(spark, tag_name="Service_Plan", vertex_id_field="planID")
    write_vertices_to_nebula(spark, tag_name="Application", vertex_id_field="appID")
    
    # Load all edge data
    write_edges_to_nebula(spark, edge_name="HAS_DEVICE")
    write_edges_to_nebula(spark, edge_name="HAS_PHONE_NUMBER")
    write_edges_to_nebula(spark, edge_name="SUBSCRIBES_TO")
    write_edges_to_nebula(spark, edge_name="MAKES_CALL", rank_field="rank")
    # write_edges_to_nebula(spark, edge_name="SENDS_SMS", rank_field="rank")
    write_edges_to_nebula(spark, edge_name="USES_DATA", rank_field="rank")
    write_edges_to_nebula(spark, edge_name="USES_APP", rank_field="rank")

    print("\nNebulaGraph data loading job has finished successfully!")
    spark.stop()

if __name__ == "__main__":
    main()
