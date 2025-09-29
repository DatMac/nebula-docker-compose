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
NUM_CUSTOMERS = 1_000_000
TARGET_STRUCTURAL_PAIR_RATIO = 80.0
HDFS_BASE_PATH = "hdfs://namenode:8020/telecom"
CHECKPOINT_DIR = "hdfs://namenode:8020/tmp/spark_checkpoints"

EDGE_TYPE_PROBABILITIES = {
    "CALL": 0.9,        
    "TRANSACTION": 0.8, 
    "SEND_SMS": 0.3     
}


def generate_customers_df(spark, num_customers):
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

    initial_rdd = spark.sparkContext.parallelize(range(num_customers), 100)
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


def generate_structural_core_edges(spark, nodes_df, num_hubs=1_000_000, edges_per_node=10):
    print(f"  -> Generating a structural core with {num_hubs} hubs.")
    nodes_df.cache()
    hub_nodes = nodes_df.sample(False, num_hubs / nodes_df.count()).limit(num_hubs).collect()
    hub_ids = [row.vertex_id for row in hub_nodes]
    hubs_bc = spark.sparkContext.broadcast(hub_ids)

    def connect_to_hub(iterator):
        local_hubs = hubs_bc.value
        if not local_hubs:
            return
        for row in iterator:
            for _ in range(edges_per_node):
                chosen_hub = random.choice(local_hubs)
                if row.vertex_id != chosen_hub:
                    yield (row.vertex_id, chosen_hub)

    core_edges_rdd = nodes_df.rdd.mapPartitions(connect_to_hub)
    return core_edges_rdd.toDF(["src", "dst"])


def densify_graph_distributively(spark, existing_edges_df, nodes_df, num_edges_to_add):
    if num_edges_to_add <= 0:
        return existing_edges_df
    print(f"  -> Robustly adding {num_edges_to_add} more random edges.")
    num_candidates_to_generate = int(num_edges_to_add * 1.3)
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
    sample_fraction = min(1.0, num_candidates_to_generate / num_potential)
    random_edge_candidates = potential_edges.sample(False, sample_fraction)
    new_unique_candidates = random_edge_candidates.join(
        existing_edges_df,
        (random_edge_candidates.src == existing_edges_df.src) & (random_edge_candidates.dst == existing_edges_df.dst),
        "left_anti"
    )
    additional_edges_df = new_unique_candidates.distinct().limit(num_edges_to_add)
    return existing_edges_df.union(additional_edges_df)


if __name__ == "__main__":
    start_time = time.time()
    spark = SparkSession.builder.appName("ScalableMultiTypeGraphGenerator").getOrCreate()
    sc = spark.sparkContext
    sc.setCheckpointDir(CHECKPOINT_DIR)

    print(f"--- Generating {NUM_CUSTOMERS} customer nodes ---")
    customers_df = generate_customers_df(spark, NUM_CUSTOMERS)
    customers_df.cache()
    customers_df.write.mode("overwrite").parquet(f"{HDFS_BASE_PATH}/vertices/Customer")
    print("  -> Customer vertex data written to HDFS as Parquet.")

    # --- Step 1: Generate the Structural Canvas of Unique Pairs ---
    print("\n--- Generating structural canvas of unique (src, dst) pairs ---")
    structural_edges_df = generate_structural_core_edges(spark, customers_df.select("vertex_id"))
    structural_edges_df.cache()

    num_core_edges = structural_edges_df.count()
    target_num_pairs = int(NUM_CUSTOMERS * TARGET_STRUCTURAL_PAIR_RATIO)
    num_pairs_to_add = target_num_pairs - num_core_edges

    print(f"  -> Target number of unique pairs: {target_num_pairs}.")
    structural_pair_canvas_df = densify_graph_distributively(
        spark, structural_edges_df, customers_df.select("vertex_id"), num_pairs_to_add
    )
    structural_pair_canvas_df = structural_pair_canvas_df.checkpoint()
    print(f"  -> Generated a canvas of {structural_pair_canvas_df.count()} unique pairs.")

    # --- Step 2: "Paint" the Canvas with Independent Edge Types ---
    print("\n--- Assigning multiple, independent edge types to the structural canvas ---")
    
    # Add a boolean column for each edge type based on its probability
    expanded_edges_df = structural_pair_canvas_df \
        .withColumn("has_call", F.rand() < EDGE_TYPE_PROBABILITIES["CALL"]) \
        .withColumn("has_sms", F.rand() < EDGE_TYPE_PROBABILITIES["SEND_SMS"]) \
        .withColumn("has_transaction", F.rand() < EDGE_TYPE_PROBABILITIES["TRANSACTION"])
        
    expanded_edges_df.cache()

    # --- Step 3: Create and Save Final DataFrames for Each Type ---
    
    # CALL edges
    calls_df = expanded_edges_df.where(F.col("has_call")) \
        .withColumn("number_of_calls", (F.rand() * 100 + 1).cast(IntegerType())) \
        .withColumn("total_call_duration", (F.rand() * 36000 + 60).cast(IntegerType())) \
        .withColumn("total_call_charge", F.rand() * 500 + 1.0) \
        .select(F.col("src").alias("src_id"), F.col("dst").alias("dst_id"), "number_of_calls", "total_call_duration", "total_call_charge")

    # SEND_SMS edges
    sms_df = expanded_edges_df.where(F.col("has_sms")) \
        .withColumn("number_of_sms_sends", (F.rand() * 500 + 1).cast(IntegerType())) \
        .withColumn("total_sms_charge", F.rand() * 100 + 0.5) \
        .select(F.col("src").alias("src_id"), F.col("dst").alias("dst_id"), "number_of_sms_sends", "total_sms_charge")

    # TRANSACTION edges
    transactions_df = expanded_edges_df.where(F.col("has_transaction")) \
        .withColumn("number_of_transaction", (F.rand() * 50 + 1).cast(IntegerType())) \
        .withColumn("total_number_of_transaction", F.col("number_of_transaction")) \
        .withColumn("total_transaction_charge", F.rand() * 10000 + 10.0) \
        .select(F.col("src").alias("src_id"), F.col("dst").alias("dst_id"), "number_of_transaction", "total_number_of_transaction", "total_transaction_charge")

    print(f"\n--- Writing multi-type edge data to HDFS as Parquet ---")
    calls_df.write.mode("overwrite").parquet(f"{HDFS_BASE_PATH}/edges/CALL")
    sms_df.write.mode("overwrite").parquet(f"{HDFS_BASE_PATH}/edges/SEND_SMS")
    transactions_df.write.mode("overwrite").parquet(f"{HDFS_BASE_PATH}/edges/TRANSACTION")
    
    print("  -> Finished writing all edge data.")
    
    total_edges = calls_df.count() + sms_df.count() + transactions_df.count()
    print(f"\nApproximated final total edge count: {total_edges}")

    end_time = time.time()
    print(f"Data generation complete. Total time: {end_time - start_time:.2f} seconds.")

    spark.stop()
