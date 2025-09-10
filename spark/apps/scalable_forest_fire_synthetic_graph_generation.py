import random
import uuid
import time

from faker import Faker
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DateType,
    IntegerType, DoubleType
)

# --- Configuration ---
NUM_CUSTOMERS = 10_000_000
TARGET_EDGE_NODE_RATIO = 8
HDFS_BASE_PATH = "hdfs://namenode:8020/telecom"
CHECKPOINT_DIR = "hdfs://namenode:8020/tmp/spark_checkpoints"


def generate_customers_df(spark, num_customers):
    """Generates a DataFrame of customer nodes with Parquet-compatible column names."""
    def generate_partition(iterator):
        fake = Faker()
        partition_data = []
        for i in iterator:
            issue_date = fake.date_this_decade()
            partition_data.append((
                str(uuid.uuid4()), f"CUST-{i+1:08d}", random.choice(["Prepaid", "Postpaid"]),
                fake.name(), fake.date_of_birth(minimum_age=18, maximum_age=80),
                random.choice(["Male", "Female"]), fake.country(), "Active", fake.address(),
                "Generated customer data", random.choice(["Passport", "ID Card"]),
                str(random.randint(100000000, 999999999)), fake.city(), issue_date,
                fake.future_date(end_date="+10y")
            ))
        return iter(partition_data)

    initial_rdd = spark.sparkContext.parallelize(range(num_customers), 100) # More partitions for larger data
    customer_rdd = initial_rdd.mapPartitions(generate_partition)

    customer_schema = StructType([
        StructField("vertex_id", StringType(), False), StructField("cust_id", StringType(), False),
        StructField("cust_type", StringType(), True), StructField("name", StringType(), True),
        StructField("birth_date", DateType(), True), StructField("sex", StringType(), True),
        StructField("nationality", StringType(), True), StructField("status", StringType(), True),
        StructField("address", StringType(), True), StructField("description", StringType(), True),
        StructField("id_type", StringType(), True), StructField("id_no", StringType(), True),
        StructField("id_issue_place", StringType(), True), StructField("id_issue_date", DateType(), True),
        StructField("id_expire_date", DateType(), True),
    ])
    return spark.createDataFrame(customer_rdd, customer_schema)


def generate_structural_core_edges(spark, nodes_df, num_hubs=10000, edges_per_node=3):
    """
    Creates a base set of edges using a hub-and-spoke model in a distributed way.
    This avoids collecting all nodes to the driver.
    """
    print(f"  -> Generating a structural core with {num_hubs} hubs.")
    nodes_df.cache()

    # 1. Sample a small number of nodes to act as hubs.
    # .sample() is a distributed operation. .collect() is then safe on this tiny sample.
    hub_nodes = nodes_df.sample(False, num_hubs / nodes_df.count()).limit(num_hubs).collect()
    hub_ids = [row.vertex_id for row in hub_nodes]
    
    # Broadcast the small list of hub IDs to all executors
    hubs_bc = spark.sparkContext.broadcast(hub_ids)

    # 2. Each node connects to a random hub
    def connect_to_hub(iterator):
        local_hubs = hubs_bc.value
        if not local_hubs:
            return
        for row in iterator:
            for _ in range(edges_per_node):
                chosen_hub = random.choice(local_hubs)
                # Ensure a node doesn't connect to itself if it's a hub
                if row.vertex_id != chosen_hub:
                    yield (row.vertex_id, chosen_hub)

    core_edges_rdd = nodes_df.rdd.mapPartitions(connect_to_hub)
    return core_edges_rdd.toDF(["src", "dst"])


def densify_graph_distributively(spark, existing_edges_df, nodes_df, num_edges_to_add):
    """Adds a specific number of random edges to a graph in a fully distributed way."""
    if num_edges_to_add <= 0:
        return existing_edges_df

    print(f"  -> Adding {num_edges_to_add} more random edges distributively.")
    num_buckets = 200

    src_nodes = nodes_df.withColumn("bucket_id", (F.rand() * num_buckets).cast("int"))
    dst_nodes = nodes_df.withColumn("bucket_id", (F.rand() * num_buckets).cast("int")) \
                        .withColumnRenamed("vertex_id", "dst_vertex_id")

    potential_edges = src_nodes.join(dst_nodes, "bucket_id") \
                               .where(F.col("vertex_id") != F.col("dst_vertex_id")) \
                               .select(F.col("vertex_id").alias("src"), F.col("dst_vertex_id").alias("dst"))

    num_potential = potential_edges.count()
    if num_potential == 0:
        return existing_edges_df

    sample_fraction = min(1.0, num_edges_to_add / num_potential * 1.2) # Add buffer for distinct
    additional_edges_df = potential_edges.sample(False, sample_fraction).limit(num_edges_to_add)

    return existing_edges_df.union(additional_edges_df).distinct()

if __name__ == "__main__":
    start_time = time.time()
    spark = SparkSession.builder.appName("ScalableGraphGenerator").getOrCreate()
    sc = spark.sparkContext
    sc.setCheckpointDir(CHECKPOINT_DIR)

    print(f"--- Generating {NUM_CUSTOMERS} customer nodes ---")
    customers_df = generate_customers_df(spark, NUM_CUSTOMERS)
    customers_df.cache()
    customers_df.write.mode("overwrite").parquet(f"{HDFS_BASE_PATH}/vertices/Customer")
    print("  -> Customer vertex data written to HDFS as Parquet.")

    # --- Step 1: Generate a non-random structural core (SCALABLE) ---
    print("\n--- Generating base graph structure using a scalable hub-spoke model ---")
    # This creates about 1 edge per node, giving us an initial structure.
    structural_edges_df = generate_structural_core_edges(spark, customers_df.select("vertex_id"))
    structural_edges_df.cache()

    # --- Step 2: Densify Graph to Meet Target Ratio (SCALABLE) ---
    num_core_edges = structural_edges_df.count()
    target_num_edges = NUM_CUSTOMERS * TARGET_EDGE_NODE_RATIO
    num_edges_to_add = target_num_edges - num_core_edges

    print(f"\n--- Densifying graph ---")
    print(f"  -> Structural core generated {num_core_edges} edges.")
    print(f"  -> Target total edges: {target_num_edges}.")

    combined_edges_df = densify_graph_distributively(spark, structural_edges_df, customers_df.select("vertex_id"), num_edges_to_add)
    combined_edges_df = combined_edges_df.checkpoint() # Checkpoint after the heavy lifting

    print("\n--- Assigning edge types and properties with new frequency distribution ---")
    edges_with_type_df = combined_edges_df.withColumn(
        "edge_type",
        F.when(F.rand() < 0.45, "CALL")
         .when(F.rand() < 0.85, "TRANSACTION")
         .otherwise("SEND_SMS")
    )
    edges_with_type_df.cache()
    final_count = edges_with_type_df.count()
    print(f"  -> Final edge count: {final_count}")

    calls_df = edges_with_type_df.where(F.col("edge_type") == "CALL") \
        .withColumn("number_of_calls", (F.rand() * 100 + 1).cast(IntegerType())) \
        .withColumn("total_call_duration", (F.rand() * 36000 + 60).cast(IntegerType())) \
        .withColumn("total_call_charge", F.rand() * 500 + 1.0) \
        .select(F.col("src").alias("src_id"), F.col("dst").alias("dst_id"), "number_of_calls", "total_call_duration", "total_call_charge")

    sms_df = edges_with_type_df.where(F.col("edge_type") == "SEND_SMS") \
        .withColumn("number_of_sms_sends", (F.rand() * 500 + 1).cast(IntegerType())) \
        .withColumn("total_sms_charge", F.rand() * 100 + 0.5) \
        .select(F.col("src").alias("src_id"), F.col("dst").alias("dst_id"), "number_of_sms_sends", "total_sms_charge")

    transactions_df = edges_with_type_df.where(F.col("edge_type") == "TRANSACTION") \
        .withColumn("number_of_transaction", (F.rand() * 50 + 1).cast(IntegerType())) \
        .withColumn("total_number_of_transaction", F.col("number_of_transaction")) \
        .withColumn("total_transaction_charge", F.rand() * 10000 + 10.0) \
        .select(F.col("src").alias("src_id"), F.col("dst").alias("dst_id"), "number_of_transaction", "total_number_of_transaction", "total_transaction_charge")

    print(f"\n--- Writing edge data to HDFS as Parquet ---")
    calls_df.write.mode("overwrite").parquet(f"{HDFS_BASE_PATH}/edges/CALL")
    sms_df.write.mode("overwrite").parquet(f"{HDFS_BASE_PATH}/edges/SEND_SMS")
    transactions_df.write.mode("overwrite").parquet(f"{HDFS_BASE_PATH}/edges/TRANSACTION")
    print("  -> Finished writing all edge data.")

    end_time = time.time()
    print(f"\nData generation complete. Total time: {end_time - start_time:.2f} seconds.")

    spark.stop()
