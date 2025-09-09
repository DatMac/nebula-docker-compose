# spark/apps/test_nebula_connector.py

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

def main():
    # The Spark master URL points to the service name in docker-compose
    spark = SparkSession.builder \
        .appName("NebulaSparkConnectorTest") \
        .master("spark://spark-master:7077") \
        .getOrCreate()

    print("Spark Session created successfully.")

    # ===============================================================
    # TEST 1: Read Vertices from NebulaGraph
    # ===============================================================
    print("\n--- [Test 1] Reading 'player' vertices from NebulaGraph ---")
    
    df_vertices = spark.read.format("com.vesoft.nebula.connector.NebulaDataSource") \
                            .option("type", "vertex") \
                            .option("operateType", "read") \
                            .option("label", "player") \
                            .option("returnCols", "name,age") \
                            .option("metaAddress", "metad0:9559") \
                            .option("spaceName", "spark_test") \
                            .option("user", "root") \
                            .option("passwd", "nebula") \
                            .option("partitionNumber", 1) \
                            .load()
    
    print("Schema of vertex DataFrame:")
    df_vertices.printSchema()
    print("Data read from 'player' vertices:")
    df_vertices.show()

    # ===============================================================
    # TEST 2: Read Edges from NebulaGraph
    # ===============================================================
    print("\n--- [Test 2] Reading 'follow' edges from NebulaGraph ---")
    
    df_edges = spark.read.format("com.vesoft.nebula.connector.NebulaDataSource") \
         .option("type", "edge") \
         .option("operateType", "read") \
         .option("label", "follow") \
         .option("returnCols", "degree") \
         .option("metaAddress", "metad0:9559") \
         .option("spaceName", "spark_test") \
         .option("user", "root") \
         .option("passwd", "nebula") \
         .option("partitionNumber", 1) \
         .load()

    print("Schema of edge DataFrame:")
    df_edges.printSchema()
    print("Data read from 'follow' edges:")
    df_edges.show()


    # ===============================================================
    # TEST 3: Write New Vertices to NebulaGraph
    # ===============================================================
    print("\n--- [Test 3] Writing new 'player' vertices to NebulaGraph ---")

    new_players_data = [(103, "Michael Jordan", 59), (104, "LeBron James", 38)]
    schema = StructType([
        StructField("player_id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("age", IntegerType(), True)
    ])
    
    df_new_players = spark.createDataFrame(data=new_players_data, schema=schema)
    
    print("DataFrame to be written as vertices:")
    df_new_players.show()

    df_new_players.write.format("com.vesoft.nebula.connector.NebulaDataSource") \
         .option("type", "vertex") \
         .option("operateType", "write") \
         .option("label", "player") \
         .option("vidPolicy", "") \
         .option("vertexField", "player_id") \
         .option("batch", 1) \
         .option("writeMode", "insert") \
         .option("metaAddress", "metad0:9559,metad1:9559,metad2:9559") \
         .option("graphAddress", "graphd:9669,graphd1:9669,graphd2:9669") \
         .option("spaceName", "spark_test") \
         .option("user", "root") \
         .option("passwd", "nebula") \
         .save()

    print("Successfully wrote new vertices. Check NebulaGraph to verify.")

    # ===============================================================
    # TEST 4: Write New Edges to NebulaGraph
    # ===============================================================
    print("\n--- [Test 4] Writing new 'follow' edges to NebulaGraph ---")

    new_follow_data = [(103, 104, 99.9)]
    schema = StructType([
        StructField("source_id", IntegerType(), True),
        StructField("dest_id", IntegerType(), True),
        StructField("degree", DoubleType(), True)
    ])

    df_new_follows = spark.createDataFrame(data=new_follow_data, schema=schema)

    print("DataFrame to be written as edges:")
    df_new_follows.show()
    
    df_new_follows.write.format("com.vesoft.nebula.connector.NebulaDataSource") \
        .option("type", "edge") \
        .option("operateType", "write") \
        .option("srcPolicy", "") \
        .option("dstPolicy", "") \
        .option("metaAddress", "metad0:9559,metad1:9559,metad2:9559") \
        .option("graphAddress", "graphd:9669,graphd1:9669,graphd2:9669") \
        .option("user", "root") \
        .option("passwd", "nebula") \
        .option("spaceName", "spark_test") \
        .option("label", "follow") \
        .option("srcVertexField", "source_id") \
        .option("dstVertexField", "dest_id") \
        .option("rankField", "") \
        .option("writeMode", "insert") \
        .option("batch", 1) \
        .save()
        
    print("Successfully wrote new edges. Check NebulaGraph to verify.")

    spark.stop()

if __name__ == "__main__":
    main()
