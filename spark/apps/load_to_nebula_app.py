# spark/apps/load_hdfs_to_nebula.py

import re
from pyspark.sql import SparkSession, DataFrame

# --- Configuration ---
HDFS_BASE_PATH = "hdfs://namenode:8020/tmp"
NEBULA_META_ADDRESS = "metad0:9559,metad1:9559,metad2:9559"
NEBULA_GRAPH_ADDRESS = "graphd:9669,graphd1:9669,graphd2:9669"
NEBULA_SPACE = "movie_kg"
NEBULA_USER = "root"
NEBULA_PASSWORD = "nebula"
WRITE_BATCH_SIZE = 512

def clean_dataframe_columns(df: DataFrame) -> DataFrame:
    """
    Cleans the column names of a DataFrame read from our generated CSV.
    Removes the type hints like ':string' or ':ID(person)'.
    Example: 'personID:ID(person)' becomes 'personID'.
    """
    new_columns = [re.split(r'[:(]', col)[0] for col in df.columns]
    return df.toDF(*new_columns)

def write_vertices_to_nebula(spark: SparkSession, tag_name: str, vertex_id_field: str):
    """
    Reads vertex data from HDFS, cleans it, and writes it to NebulaGraph.
    
    :param spark: The active SparkSession.
    :param tag_name: The name of the Nebula tag (e.g., 'person').
    :param vertex_id_field: The name of the column containing the vertex ID.
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
        .option("writeMode", "insert") \
        .option("batch", WRITE_BATCH_SIZE) \
        .save()
        
    print(f"Successfully wrote vertices for tag '{tag_name}'.")

def write_edges_to_nebula(spark: SparkSession, edge_name: str, src_id_field: str = "src", dst_id_field: str = "dst"):
    """
    Reads edge data from HDFS, cleans it, and writes it to NebulaGraph.

    :param spark: The active SparkSession.
    :param edge_name: The name of the Nebula edge (e.g., 'acted_in').
    :param src_id_field: The name of the column for the source vertex ID.
    :param dst_id_field: The name of the column for the destination vertex ID.
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
    df_clean.write.format("com.vesoft.nebula.connector.NebulaDataSource") \
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
        .option("rankField", "") \
        .option("writeMode", "insert") \
        .option("batch", WRITE_BATCH_SIZE) \
        .save()
        
    print(f"Successfully wrote edges for type '{edge_name}'.")


def main():
    # Initialize Spark Session
    # Note: Spark connector JARs need to be specified in the spark-submit command
    spark = SparkSession.builder \
        .appName("HDFS to NebulaGraph Loader") \
        .master("spark://spark-master:7077") \
        .getOrCreate()

    print("Spark Session created. Starting data load process...")

    # Load all vertex data
    write_vertices_to_nebula(spark, tag_name="person", vertex_id_field="personID")
    write_vertices_to_nebula(spark, tag_name="movie", vertex_id_field="movieID")
    write_vertices_to_nebula(spark, tag_name="genre", vertex_id_field="genreID")
    write_vertices_to_nebula(spark, tag_name="user", vertex_id_field="userID")
    
    # Load all edge data
    write_edges_to_nebula(spark, edge_name="acted_in")
    write_edges_to_nebula(spark, edge_name="directed")
    write_edges_to_nebula(spark, edge_name="belongs_to")
    write_edges_to_nebula(spark, edge_name="rated")

    print("\nData loading job finished successfully!")
    spark.stop()

if __name__ == "__main__":
    main()
