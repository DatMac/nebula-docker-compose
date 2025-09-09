import random
import uuid
from datetime import datetime, timedelta
import time

from faker import Faker
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, DateType,
    IntegerType, DoubleType
)

# --- Configuration ---
NUM_CUSTOMERS = 10000
AVG_EDGES_PER_CUSTOMER = 5
HDFS_BASE_PATH = "hdfs://namenode:8020/telecom"

def generate_customers_df(spark, num_customers):
    """Generates a DataFrame of customer nodes with a separate vertex_id."""

    def generate_partition(iterator):
        fake = Faker()
        for i in iterator:
            yield (
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
                fake.date_this_decade(),
                fake.date_this_decade(after_now=True)
            )

    rdd = spark.sparkContext.parallelize(range(num_customers), numSlices=10)
    customer_rdd = rdd.mapPartitions(generate_partition)

    customer_schema = StructType([
        StructField("vertex_id:ID(Customer)", StringType(), False),
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

def generate_edges_df(spark, customer_vertex_ids_bc, edge_type):
    """Generates a DataFrame for a specific type of edge using vertex_ids."""
    num_edges = int(len(customer_vertex_ids_bc.value) * AVG_EDGES_PER_CUSTOMER * random.uniform(0.5, 1.5))

    def generate_partition(iterator):
        local_customer_ids = customer_vertex_ids_bc.value
        for _ in iterator:
            source_id, dest_id = random.sample(local_customer_ids, 2)
            if edge_type == "CALL":
                yield (source_id, dest_id, random.randint(1, 100), random.randint(60, 36000), round(random.uniform(1.0, 500.0), 2))
            elif edge_type == "SEND_SMS":
                yield (source_id, dest_id, random.randint(1, 500), round(random.uniform(0.5, 100.0), 2))
            elif edge_type == "TRANSACTION": 
                yield (source_id, dest_id, random.randint(1, 50), random.randint(1, 50), round(random.uniform(10.0, 10000.0), 2))
    rdd = spark.sparkContext.parallelize(range(num_edges), numSlices=10)
    edge_rdd = rdd.mapPartitions(generate_partition)

    if edge_type == "CALL":
        schema = StructType([
            StructField(":START_ID(Customer)", StringType(), False),
            StructField(":END_ID(Customer)", StringType(), False),
            StructField("number_of_calls", IntegerType(), True),
            StructField("total_call_duration", IntegerType(), True),
            StructField("total_call_charge", DoubleType(), True)
        ])
    elif edge_type == "SEND_SMS":
        schema = StructType([
            StructField(":START_ID(Customer)", StringType(), False),
            StructField(":END_ID(Customer)", StringType(), False),
            StructField("number_of_sms_sends", IntegerType(), True),
            StructField("total_sms_charge", DoubleType(), True)
        ])
    elif edge_type == "TRANSACTION":
        schema = StructType([
            StructField(":START_ID(Customer)", StringType(), False),
            StructField(":END_ID(Customer)", StringType(), False),
            StructField("number_of_transaction", IntegerType(), True),
            StructField("total_number_of_transaction", IntegerType(), True),
            StructField("total_transaction_charge", DoubleType(), True)
        ])
    else:
        raise ValueError(f"Unknown edge type: {edge_type}")

    return spark.createDataFrame(edge_rdd, schema)

if __name__ == "__main__":
    start_time = time.time()
    spark = SparkSession.builder.appName("SyntheticGraphGenerator").getOrCreate()
    sc = spark.sparkContext

    print(f"--- Generating {NUM_CUSTOMERS} customer nodes ---")
    customers_df = generate_customers_df(spark, NUM_CUSTOMERS)
    customers_df.cache()

    # Collect the new vertex_id for creating edges
    customer_vertex_ids = [row[0] for row in customers_df.select("vertex_id:ID(Customer)").collect()]
    customer_vertex_ids_bc = sc.broadcast(customer_vertex_ids)

    print("--- Generating 'call' edges ---")
    calls_df = generate_edges_df(spark, customer_vertex_ids_bc, "CALL")

    print("--- Generating 'send_sms' edges ---")
    sms_df = generate_edges_df(spark, customer_vertex_ids_bc, "SEND_SMS")

    print("--- Generating 'Transaction' edges ---")
    transactions_df = generate_edges_df(spark, customer_vertex_ids_bc, "TRANSACTION")

    print(f"\n--- Writing data to HDFS at {HDFS_BASE_PATH} ---")
    customers_df.write.mode("overwrite").option("header", "true").csv(f"{HDFS_BASE_PATH}/vertices/Customer")
    calls_df.write.mode("overwrite").option("header", "true").csv(f"{HDFS_BASE_PATH}/edges/CALL")
    sms_df.write.mode("overwrite").option("header", "true").csv(f"{HDFS_BASE_PATH}/edges/SEND_SMS")
    transactions_df.write.mode("overwrite").option("header", "true").csv(f"{HDFS_BASE_PATH}/edges/TRANSACTION")

    end_time = time.time()
    print(f"\nData generation complete. Total time: {end_time - start_time:.2f} seconds.")

    spark.stop()
