import random
import uuid
from datetime import datetime, timedelta
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

# --- Forest Fire Model Parameters ---
FORWARD_BURN_PROBABILITY = 0.75
NEW_NODES_PER_ITERATION = 2_000_000

def generate_customers_df(spark, num_customers):
    """Generates a DataFrame of customer nodes with Parquet-compatible column names."""
    def generate_partition(iterator):
        fake = Faker()
        partition_data = []
        for i in iterator:
            issue_date = fake.date_this_decade()
            partition_data.append((
                str(uuid.uuid4()),
                f"CUST-{i+1:08d}",
                random.choice(["Prepaid", "Postpaid"]),
                fake.name(),
                fake.date_of_birth(minimum_age=18, maximum_age=80),
                random.choice(["Male", "Female"]),
                fake.country(),
                "Active",
                fake.address(),
                "Generated customer data",
                random.choice(["Passport", "ID Card"]),
                str(random.randint(100000000, 999999999)),
                fake.city(),
                issue_date,
                fake.future_date(end_date="+10y")
            ))
        return iter(partition_data)

    initial_rdd = spark.sparkContext.parallelize(range(num_customers), 10)
    customer_rdd = initial_rdd.mapPartitions(generate_partition)

    customer_schema = StructType([
        StructField("vertex_id", StringType(), False),
        StructField("cust_id", StringType(), False),
        StructField("cust_type", StringType(), True),
        StructField("name", StringType(), True),
        StructField("birth_date", DateType(), True),
        StructField("sex", StringType(), True),
        StructField("nationality", StringType(), True),
        StructField("status", StringType(), True),
        StructField("address", StringType(), True),
        StructField("description", StringType(), True),
        StructField("id_type", StringType(), True),
        StructField("id_no", StringType(), True),
        StructField("id_issue_place", StringType(), True),
        StructField("id_issue_date", DateType(), True),
        StructField("id_expire_date", DateType(), True),
    ])

    return spark.createDataFrame(customer_rdd, customer_schema)


def generate_graph_with_forest_fire(spark, customers_df):
    """
    Generates a graph using a batched Forest Fire model simulation.
    """
    # MODIFIED: Select the new, clean column name.
    all_customer_ids = [row[0] for row in customers_df.select("vertex_id").collect()]
    if len(all_customer_ids) < 2:
        raise ValueError("Cannot generate a graph with fewer than 2 customers.")

    existing_nodes = set(all_customer_ids[:2])
    nodes_to_add = all_customer_ids[2:]
    
    edge_schema = StructType([
        StructField("src", StringType(), False),
        StructField("dst", StringType(), False)
    ])
    edges_df = spark.createDataFrame([(all_customer_ids[0], all_customer_ids[1])], edge_schema)
    edges_df = edges_df.checkpoint()

    num_iterations = (len(nodes_to_add) + NEW_NODES_PER_ITERATION - 1) // NEW_NODES_PER_ITERATION

    for i in range(num_iterations):
        print(f"--- Forest Fire Iteration {i+1}/{num_iterations} ---")
        start_index = i * NEW_NODES_PER_ITERATION
        end_index = start_index + NEW_NODES_PER_ITERATION
        new_nodes_batch = nodes_to_add[start_index:end_index]

        if not new_nodes_batch:
            continue
            
        existing_nodes_list = list(existing_nodes)
        ambassador_assignments = []
        for node_id in new_nodes_batch:
            ambassador = random.choice(existing_nodes_list)
            ambassador_assignments.append((node_id, ambassador))

        ambassador_edges_df = spark.createDataFrame(ambassador_assignments, ["src", "dst"])
        ambassadors_df = ambassador_edges_df.select(F.col("dst").alias("ambassador_id")).distinct()
        
        neighbor_edges_1 = edges_df.join(ambassadors_df, F.col("src") == F.col("ambassador_id")).select(F.col("dst").alias("neighbor_id"), F.col("ambassador_id"))
        neighbor_edges_2 = edges_df.join(ambassadors_df, F.col("dst") == F.col("ambassador_id")).select(F.col("src").alias("neighbor_id"), F.col("ambassador_id"))
        all_neighbors_df = neighbor_edges_1.union(neighbor_edges_2).distinct()

        potential_fire_edges_df = ambassador_edges_df.join(
            all_neighbors_df, F.col("dst") == F.col("ambassador_id")
        ).select(F.col("src"), F.col("neighbor_id").alias("dst"))

        fire_edges_df = potential_fire_edges_df.where(F.rand() < FORWARD_BURN_PROBABILITY)
        new_edges_df = ambassador_edges_df.union(fire_edges_df)
        
        old_edges_df = edges_df
        edges_df = old_edges_df.union(new_edges_df).distinct()
        edges_df = edges_df.checkpoint()
        old_edges_df.unpersist()

        existing_nodes.update(new_nodes_batch)

    return edges_df

def densify_graph_distributively(spark, existing_edges_df, nodes_df, num_edges_to_add):
    """
    Adds a specific number of random edges to a graph in a fully distributed way,
    avoiding .collect().
    """
    if num_edges_to_add <= 0:
        return existing_edges_df

    print(f"  -> Adding {num_edges_to_add} more random edges distributively.")

    # A higher number of buckets reduces the size of each join partition.
    # A good starting point is sqrt(total_nodes / some_factor)
    num_buckets = 200

    # Create source and destination views of the nodes, each with a random bucket
    src_nodes = nodes_df.withColumn("bucket_id", (F.rand() * num_buckets).cast("int"))
    dst_nodes = nodes_df.withColumn("bucket_id", (F.rand() * num_buckets).cast("int")) \
                        .withColumnRenamed("vertex_id", "dst_vertex_id")

    # Join nodes within the same bucket to create potential edges
    # This avoids a full N x N cross-join
    potential_edges = src_nodes.join(dst_nodes, "bucket_id") \
                               .where(F.col("vertex_id") != F.col("dst_vertex_id")) \
                               .select(F.col("vertex_id").alias("src"), F.col("dst_vertex_id").alias("dst"))

    # To get the exact number of edges, we need to sample.
    # First, count how many potential edges our bucketing created.
    num_potential = potential_edges.count()
    if num_potential == 0:
        return existing_edges_df

    # Calculate the fraction needed to get the target number of edges
    sample_fraction = num_edges_to_add / num_potential

    # Add a random number and filter by the fraction to get a sample
    additional_edges_df = potential_edges.withColumn("rand", F.rand()).where(F.col("rand") < sample_fraction).select("src", "dst")

    # Union the original structural edges with the new random ones
    return existing_edges_df.union(additional_edges_df).distinct()

if __name__ == "__main__":
    start_time = time.time()
    spark = SparkSession.builder.appName("ForestFireGraphGenerator").getOrCreate()
    sc = spark.sparkContext
    sc.setCheckpointDir(CHECKPOINT_DIR)

    print(f"--- Generating {NUM_CUSTOMERS} customer nodes ---")
    customers_df = generate_customers_df(spark, NUM_CUSTOMERS)
    customers_df.cache()
    
    customers_df.write.mode("overwrite").parquet(f"{HDFS_BASE_PATH}/vertices/Customer")
    print("  -> Customer vertex data written to HDFS as Parquet.")

    print("\n--- Generating base graph structure using Forest Fire model ---")
    structural_edges_df = generate_graph_with_forest_fire(spark, customers_df)
    structural_edges_df.cache()
    
    num_ff_edges = structural_edges_df.count()
    target_num_edges = NUM_CUSTOMERS * TARGET_EDGE_NODE_RATIO
    num_edges_to_add = target_num_edges - num_ff_edges
    
    print(f"\n--- Densifying graph ---")
    print(f"  -> Forest Fire generated {num_ff_edges} edges.")
    print(f"  -> Target total edges: {target_num_edges}.")
    
    combined_edges_df = densify_graph_distributively(spark, structural_edges_df, customers_df.select("vertex_id"), num_edges_to_add)
    combined_edges_df.cache()
    
    print("\n--- Assigning edge types and properties with new frequency distribution ---")
    edges_with_type_df = combined_edges_df.withColumn(
        "edge_type",
        F.when(F.rand() < 0.45, "CALL")
         .when(F.rand() < 0.85, "TRANSACTION")
         .otherwise("SEND_SMS")
    )
    edges_with_type_df.cache()
    print("  -> Final edge count:", edges_with_type_df.count())

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
    # MODIFIED: Write edges as Parquet.
    calls_df.write.mode("overwrite").parquet(f"{HDFS_BASE_PATH}/edges/CALL")
    sms_df.write.mode("overwrite").parquet(f"{HDFS_BASE_PATH}/edges/SEND_SMS")
    transactions_df.write.mode("overwrite").parquet(f"{HDFS_BASE_PATH}/edges/TRANSACTION")
    print("  -> Finished writing all edge data.")

    end_time = time.time()
    print(f"\nData generation complete. Total time: {end_time - start_time:.2f} seconds.")

    spark.stop()
