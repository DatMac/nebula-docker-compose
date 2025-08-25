import random
import uuid
from datetime import datetime, timedelta
import time
import sys

from faker import Faker
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, DateType, TimestampType,
    IntegerType, LongType, DoubleType, BooleanType
)


# --- Configuration ---
NUM_CUSTOMERS = 10_000
NUM_CELL_TOWERS = 500
NUM_CALLS = 200_000
NUM_SMS = 500_000
NUM_DATA_SESSIONS = 1_000_000
NUM_APP_SESSIONS = 2_000_000

# HDFS output paths
HDFS_BASE_PATH = "hdfs://namenode:8020/tmp/telecom"

# --- Predefined Catalog Data ---
SERVICE_PLANS = [
    {'plan_id': 'plan_basic_25', 'plan_name': 'Basic Talk & Text', 'monthly_fee': 25.0, 'data_allowance_gb': 2, 'voice_allowance_min': 500, 'sms_allowance': 500, 'contract_length_months': 0},
    {'plan_id': 'plan_standard_45', 'plan_name': 'Standard Unlimited', 'monthly_fee': 45.0, 'data_allowance_gb': 20, 'voice_allowance_min': -1, 'sms_allowance': -1, 'contract_length_months': 12},
    {'plan_id': 'plan_premium_70', 'plan_name': 'Premium 5G Unlimited', 'monthly_fee': 70.0, 'data_allowance_gb': -1, 'voice_allowance_min': -1, 'sms_allowance': -1, 'contract_length_months': 24},
    {'plan_id': 'plan_family_120', 'plan_name': 'Family Share Plan', 'monthly_fee': 120.0, 'data_allowance_gb': 50, 'voice_allowance_min': -1, 'sms_allowance': -1, 'contract_length_months': 24},
    {'plan_id': 'plan_data_15', 'plan_name': 'Data Only 10GB', 'monthly_fee': 15.0, 'data_allowance_gb': 10, 'voice_allowance_min': 0, 'sms_allowance': 0, 'contract_length_months': 6},
    {'plan_id': 'plan_senior_20', 'plan_name': 'Senior Talk Easy', 'monthly_fee': 20.0, 'data_allowance_gb': 2, 'voice_allowance_min': 1000, 'sms_allowance': 1000, 'contract_length_months': 0},
    {'plan_id': 'plan_business_90', 'plan_name': 'Global Business Pro', 'monthly_fee': 90.0, 'data_allowance_gb': -1, 'voice_allowance_min': -1, 'sms_allowance': -1, 'contract_length_months': 24}
]

APPLICATIONS = [
    {'app_id': 'app_yt', 'app_name': 'YouTube', 'category': 'Video Streaming'},
    {'app_id': 'app_fb', 'app_name': 'Facebook', 'category': 'Social Media'},
    {'app_id': 'app_ig', 'app_name': 'Instagram', 'category': 'Social Media'},
    {'app_id': 'app_nf', 'app_name': 'Netflix', 'category': 'Video Streaming'},
    {'app_id': 'app_gm', 'app_name': 'Google Maps', 'category': 'Navigation'},
    {'app_id': 'app_sp', 'app_name': 'Spotify', 'category': 'Music Streaming'},
    {'app_id': 'app_wa', 'app_name': 'WhatsApp', 'category': 'Messaging'},
    {'app_id': 'app_tk', 'app_name': 'TikTok', 'category': 'Social Media'},
    {'app_id': 'app_x', 'app_name': 'X (Twitter)', 'category': 'Social Media'},
    {'app_id': 'app_teams', 'app_name': 'Microsoft Teams', 'category': 'Productivity'}
]


# --- Helper Function ---
def get_random_timestamp(fake_instance, start_date="-2y"):
    return fake_instance.date_time_between(start_date=start_date, end_date='now')

# --- OPTIMIZED: Distributed Data Generation Functions ---

def generate_customers_rdd(spark, num_customers):
    """Generates customer data RDD in a distributed manner."""
    print(f"--- Generating {num_customers} Customers ---")
    
    def generate_partition(iterator):
        fake = Faker()
        for i in iterator:
            yield (
                str(uuid.uuid4()), f"CUST-{i+1:08d}", fake.first_name(), fake.last_name(), fake.date_of_birth(minimum_age=18, maximum_age=90),
                random.choice(['Male', 'Female', 'Other']), fake.street_address(), fake.city(), fake.state_abbr(), fake.zipcode(),
                fake.email(), fake.phone_number(), random.choice(['Prepaid', 'Postpaid', 'Business']),
                random.choice(['Active', 'Suspended', 'Deactivated']), get_random_timestamp(fake, "-5y"),
                random.randint(300, 850), round(random.uniform(20.0, 150.0), 2), round(random.random(), 3), random.choice([True, False])
            )

    rdd = spark.sparkContext.parallelize(range(num_customers), numSlices=9)
    return rdd.mapPartitions(generate_partition)


def generate_devices_and_numbers_rdds(spark, customer_ids_bc):
    """Generates devices and phone numbers for customers in a distributed manner."""
    print("--- Generating Devices and Phone Numbers for Customers ---")
    
    def generate_partition(iterator):
        fake = Faker()
        device_models = ['iPhone 15', 'Samsung Galaxy S25', 'Google Pixel 9', 'iPhone 14 Pro']
        manufacturers = ['Apple', 'Samsung', 'Google']
        
        for cid in iterator:
            num_assets = random.randint(1, 2)
            for j in range(num_assets):
                is_primary = (j == 0)
                device_vid = str(uuid.uuid4())
                number_vid = str(uuid.uuid4())
                
                yield {
                    'type': 'device', 'data': (
                        device_vid, f"{random.randint(10**14, 10**15 - 1)}", random.choice(device_models), random.choice(manufacturers),
                        random.choice(['iOS', 'Android']), f"{random.randint(10, 17)}.{random.randint(0, 5)}",
                        fake.date_object(end_datetime=datetime.now()), get_random_timestamp(fake, "-1y"), random.choice([True, False]), 'Active'
                    )
                }
                yield {'type': 'has_device', 'data': (cid, device_vid, get_random_timestamp(fake, "-3y"), None, is_primary)}
                yield {
                    'type': 'phone_number', 'data': (
                        number_vid, fake.phone_number(), random.randint(1, 200), 'Mobile', get_random_timestamp(fake, "-4y"), None
                    )
                }
                yield {'type': 'has_phone_number', 'data': (cid, number_vid, get_random_timestamp(fake, "-4y"))}
    
    customer_ids_rdd = spark.sparkContext.parallelize(customer_ids_bc.value, numSlices=9)
    combined_rdd = customer_ids_rdd.mapPartitions(generate_partition).cache()
    
    device_rdd = combined_rdd.filter(lambda x: x['type'] == 'device').map(lambda x: x['data'])
    phone_number_rdd = combined_rdd.filter(lambda x: x['type'] == 'phone_number').map(lambda x: x['data'])
    has_device_rdd = combined_rdd.filter(lambda x: x['type'] == 'has_device').map(lambda x: x['data'])
    has_phone_number_rdd = combined_rdd.filter(lambda x: x['type'] == 'has_phone_number').map(lambda x: x['data'])
    
    return device_rdd, phone_number_rdd, has_device_rdd, has_phone_number_rdd

def generate_cell_towers_rdd(spark, num_towers):
    """Generates cell tower data RDD in a distributed manner."""
    print(f"--- Generating {num_towers} Cell Towers ---")

    def generate_partition(iterator):
        fake = Faker()
        for i in iterator:
            capacity = random.choice([1000, 2000, 5000])
            yield (
                str(uuid.uuid4()), f"TOWER-{i:04d}", float(fake.latitude()), float(fake.longitude()), fake.city(), fake.state_abbr(),
                fake.zipcode(), random.choice(['4G', '5G']), capacity, random.randint(100, capacity - 50),
                random.choice(['Active', 'Under Maintenance'])
            )

    rdd = spark.sparkContext.parallelize(range(num_towers), numSlices=9)
    return rdd.mapPartitions(generate_partition)

def generate_relationships_rdd(spark, num_records, name, generator_func):
    """Generic function to generate relationship (edge) data RDDs."""
    print(f"--- Generating {num_records} for {name} ---")

    def generate_partition(iterator):
        fake = Faker()
        for _ in iterator:
            yield generator_func(fake)
            
    rdd = spark.sparkContext.parallelize(range(num_records), numSlices=9)
    return rdd.mapPartitions(generate_partition)


if __name__ == "__main__":
    start_time = time.time()
    spark = SparkSession.builder.appName("TelecomDataGenerator").getOrCreate()
    sc = spark.sparkContext
    print(f"Spark Session created. Starting data generation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. DEFINE EXPLICIT SCHEMAS
    print("\n--- Defining explicit schemas for DataFrames ---")
    customer_schema = StructType([
        StructField("customerID:ID(Customer)", StringType(), False), StructField("customer_id", StringType(), False), StructField("first_name", StringType(), True),
        StructField("last_name", StringType(), True), StructField("date_of_birth", DateType(), True), StructField("gender", StringType(), True),
        StructField("address", StringType(), True), StructField("city", StringType(), True), StructField("state", StringType(), True),
        StructField("zip_code", StringType(), True), StructField("email", StringType(), True), StructField("phone_number", StringType(), True),
        StructField("account_type", StringType(), True), StructField("account_status", StringType(), True), StructField("join_date", TimestampType(), True),
        StructField("credit_score", LongType(), True), StructField("arpu", DoubleType(), True), StructField("churn_risk", DoubleType(), True),
        StructField("is_high_value", BooleanType(), True)
    ])
    device_schema = StructType([
        StructField("deviceID:ID(Device)", StringType(), False), StructField("imei", StringType(), False), StructField("device_model", StringType(), True),
        StructField("manufacturer", StringType(), True), StructField("os_type", StringType(), True), StructField("os_version", StringType(), True),
        StructField("purchase_date", DateType(), True), StructField("last_seen", TimestampType(), True), StructField("is_5g_capable", BooleanType(), True),
        StructField("status", StringType(), True)
    ])
    phone_number_schema = StructType([
        StructField("phoneNumberID:ID(Phone_Number)", StringType(), False), StructField("phone_number", StringType(), False),
        StructField("country_code", LongType(), True), StructField("number_type", StringType(), True),
        StructField("activation_date", TimestampType(), True), StructField("deactivation_date", TimestampType(), True)
    ])
    cell_tower_schema = StructType([
        StructField("towerID:ID(Cell_Tower)", StringType(), False), StructField("tower_id", StringType(), False), StructField("latitude", DoubleType(), True),
        StructField("longitude", DoubleType(), True), StructField("city", StringType(), True), StructField("state", StringType(), True),
        StructField("zip_code", StringType(), True), StructField("technology", StringType(), True), StructField("capacity", LongType(), True),
        StructField("current_load", LongType(), True), StructField("status", StringType(), True)
    ])
    service_plan_schema = StructType([
        StructField("planID:ID(Service_Plan)", StringType(), False), StructField("plan_id", StringType(), False), StructField("plan_name", StringType(), True),
        StructField("monthly_fee", DoubleType(), True), StructField("data_allowance_gb", LongType(), True), StructField("voice_allowance_min", LongType(), True),
        StructField("sms_allowance", LongType(), True), StructField("contract_length_months", LongType(), True)
    ])
    application_schema = StructType([
        StructField("appID:ID(Application)", StringType(), False), StructField("app_id", StringType(), False),
        StructField("app_name", StringType(), True), StructField("category", StringType(), True)
    ])
    has_device_schema = StructType([
        StructField(":START_ID(Customer)", StringType(), False), StructField(":END_ID(Device)", StringType(), False),
        StructField("start_date", TimestampType(), True), StructField("end_date", TimestampType(), True), StructField("is_primary_device", BooleanType(), True)
    ])
    has_phone_number_schema = StructType([
        StructField(":START_ID(Customer)", StringType(), False), StructField(":END_ID(Phone_Number)", StringType(), False),
        StructField("assignment_date", TimestampType(), True)
    ])
    subscribes_to_schema = StructType([
        StructField(":START_ID(Customer)", StringType(), False), StructField(":END_ID(Service_Plan)", StringType(), False),
        StructField("subscription_date", TimestampType(), True), StructField("renewal_date", DateType(), True)
    ])
    
    # Schemas for MAKES_CALL, SENDS_SMS, USES_DATA, USES_APP
    makes_call_schema = StructType([
        StructField(":START_ID(Phone_Number)", StringType(), False),
        StructField(":END_ID(Phone_Number)", StringType(), False),
        StructField(":RANK", IntegerType(), True),
        StructField("call_timestamp", TimestampType(), True),
        StructField("duration_seconds", IntegerType(), True),
        StructField("call_type", StringType(), True),
        StructField("call_result", StringType(), True),
        StructField("start_tower_id", StringType(), True),
        StructField("end_tower_id", StringType(), True),
        StructField("jitter_ms", DoubleType(), True),
        StructField("packet_loss_percent", DoubleType(), True)
    ])

    sends_sms_schema = StructType([
        StructField(":START_ID(Phone_Number)", StringType(), False),
        StructField(":END_ID(Phone_Number)", StringType(), False),
        StructField(":RANK", IntegerType(), True),
        StructField("sms_timestamp", TimestampType(), True),
        StructField("message_length", IntegerType(), True)
    ])

    uses_data_schema = StructType([
        StructField(":START_ID(Device)", StringType(), False),
        StructField(":END_ID(Cell_Tower)", StringType(), False),
        StructField(":RANK", IntegerType(), True),
        StructField("session_start_time", TimestampType(), True),
        StructField("session_end_time", TimestampType(), True),
        StructField("data_uploaded_mb", DoubleType(), True),
        StructField("data_downloaded_mb", DoubleType(), True),
        StructField("network_type", StringType(), True)
    ])

    uses_app_schema = StructType([
        StructField(":START_ID(Device)", StringType(), False),
        StructField(":END_ID(Application)", StringType(), False),
        StructField(":RANK", IntegerType(), True),
        StructField("session_start_time", TimestampType(), True),
        StructField("duration_minutes", IntegerType(), True),
        StructField("data_consumed_mb", DoubleType(), True)
    ])

    # 2. Generate Vertex Data in a DISTRIBUTED way
    customer_rdd = generate_customers_rdd(spark, NUM_CUSTOMERS)
    customer_df = spark.createDataFrame(customer_rdd, schema=customer_schema)
    customer_df.cache()

    cell_tower_rdd = generate_cell_towers_rdd(spark, NUM_CELL_TOWERS)
    cell_tower_df = spark.createDataFrame(cell_tower_rdd, schema=cell_tower_schema)
    cell_tower_df.cache()

    service_plan_df = spark.createDataFrame([(str(uuid.uuid4()),) + tuple(item.values()) for item in SERVICE_PLANS], schema=service_plan_schema)
    application_df = spark.createDataFrame([(str(uuid.uuid4()),) + tuple(item.values()) for item in APPLICATIONS], schema=application_schema)

    print("\nExtracting vertex IDs for relationship generation...")
    customer_ids = customer_df.select("customerID:ID(Customer)").rdd.flatMap(lambda x: x).collect()
    cell_tower_ids = cell_tower_df.select("towerID:ID(Cell_Tower)").rdd.flatMap(lambda x: x).collect()
    service_plan_ids = service_plan_df.select("planID:ID(Service_Plan)").rdd.flatMap(lambda x: x).collect()
    application_ids = application_df.select("appID:ID(Application)").rdd.flatMap(lambda x: x).collect()

    customer_ids_bc = sc.broadcast(customer_ids)
    cell_tower_ids_bc = sc.broadcast(cell_tower_ids)
    service_plan_ids_bc = sc.broadcast(service_plan_ids)
    application_ids_bc = sc.broadcast(application_ids)
    
    device_rdd, phone_number_rdd, has_device_rdd, has_phone_number_rdd = generate_devices_and_numbers_rdds(spark, customer_ids_bc)
    device_df = spark.createDataFrame(device_rdd, schema=device_schema)
    phone_number_df = spark.createDataFrame(phone_number_rdd, schema=phone_number_schema)
    has_device_df = spark.createDataFrame(has_device_rdd, schema=has_device_schema)
    has_phone_number_df = spark.createDataFrame(has_phone_number_rdd, schema=has_phone_number_schema)
    device_df.cache()
    phone_number_df.cache()

    device_ids = device_df.select("deviceID:ID(Device)").rdd.flatMap(lambda x: x).collect()
    phone_number_ids = phone_number_df.select("phoneNumberID:ID(Phone_Number)").rdd.flatMap(lambda x: x).collect()
    device_ids_bc = sc.broadcast(device_ids)
    phone_number_ids_bc = sc.broadcast(phone_number_ids)

    # 3. Generate Edge/Relationship Data in a DISTRIBUTED way
    def subscribes_to_generator(fake):
        return (random.choice(customer_ids_bc.value), random.choice(service_plan_ids_bc.value), get_random_timestamp(fake), fake.date_object())
    subscribes_to_df = spark.createDataFrame(generate_relationships_rdd(spark, NUM_CUSTOMERS * 2, "SUBSCRIBES_TO", subscribes_to_generator), schema=subscribes_to_schema)

    def makes_call_generator(fake):
        src_num, dst_num = random.sample(phone_number_ids_bc.value, 2)
        return (src_num, dst_num, 0, get_random_timestamp(fake), random.randint(5, 3600), 'Voice', random.choice(['Answered', 'Missed', 'Voicemail']), random.choice(cell_tower_ids_bc.value), random.choice(cell_tower_ids_bc.value), round(random.uniform(1.0, 50.0), 2), round(random.uniform(0.0, 2.0), 3))
    makes_call_df = spark.createDataFrame(generate_relationships_rdd(spark, NUM_CALLS, "MAKES_CALL", makes_call_generator), schema=makes_call_schema)

    def sends_sms_generator(fake):
        src_num, dst_num = random.sample(phone_number_ids_bc.value, 2)
        return (src_num, dst_num, 0, get_random_timestamp(fake), random.randint(5, 160))
    sends_sms_df = spark.createDataFrame(generate_relationships_rdd(spark, NUM_SMS, "SENDS_SMS", sends_sms_generator), schema=sends_sms_schema)
    
    def uses_data_generator(fake):
        start_time = get_random_timestamp(fake)
        return (random.choice(device_ids_bc.value), random.choice(cell_tower_ids_bc.value), 0, start_time, start_time + timedelta(minutes=random.randint(1, 60)), round(random.uniform(0.1, 50.0), 3), round(random.uniform(1.0, 1000.0), 3), random.choice(['4G', '5G']))
    uses_data_df = spark.createDataFrame(generate_relationships_rdd(spark, NUM_DATA_SESSIONS, "USES_DATA", uses_data_generator), schema=uses_data_schema)

    def uses_app_generator(fake):
        return (random.choice(device_ids_bc.value), random.choice(application_ids_bc.value), 0, get_random_timestamp(fake), random.randint(1, 120), round(random.uniform(1, 500), 2))
    uses_app_df = spark.createDataFrame(generate_relationships_rdd(spark, NUM_APP_SESSIONS, "USES_APP", uses_app_generator), schema=uses_app_schema)

    # 4. Write DataFrames to HDFS
    print(f"\n--- Writing data to HDFS at {HDFS_BASE_PATH} ---")
    customer_df.write.mode("overwrite").option("header", "true").csv(f"{HDFS_BASE_PATH}/vertices/Customer")
    device_df.write.mode("overwrite").option("header", "true").csv(f"{HDFS_BASE_PATH}/vertices/Device")
    phone_number_df.write.mode("overwrite").option("header", "true").csv(f"{HDFS_BASE_PATH}/vertices/Phone_Number")
    cell_tower_df.write.mode("overwrite").option("header", "true").csv(f"{HDFS_BASE_PATH}/vertices/Cell_Tower")
    service_plan_df.write.mode("overwrite").option("header", "true").csv(f"{HDFS_BASE_PATH}/vertices/Service_Plan")
    application_df.write.mode("overwrite").option("header", "true").csv(f"{HDFS_BASE_PATH}/vertices/Application")
    print("  -> Finished writing vertex data.")
    
    has_device_df.write.mode("overwrite").option("header", "true").csv(f"{HDFS_BASE_PATH}/edges/HAS_DEVICE")
    has_phone_number_df.write.mode("overwrite").option("header", "true").csv(f"{HDFS_BASE_PATH}/edges/HAS_PHONE_NUMBER")
    subscribes_to_df.write.mode("overwrite").option("header", "true").csv(f"{HDFS_BASE_PATH}/edges/SUBSCRIBES_TO")
    makes_call_df.write.mode("overwrite").option("header", "true").csv(f"{HDFS_BASE_PATH}/edges/MAKES_CALL")
    sends_sms_df.write.mode("overwrite").option("header", "true").csv(f"{HDFS_BASE_PATH}/edges/SENDS_SMS")
    uses_data_df.write.mode("overwrite").option("header", "true").csv(f"{HDFS_BASE_PATH}/edges/USES_DATA")
    uses_app_df.write.mode("overwrite").option("header", "true").csv(f"{HDFS_BASE_PATH}/edges/USES_APP")
    print("  -> Finished writing edge data.")

    end_time = time.time()
    print(f"\nData generation complete and saved to HDFS. Total time: {end_time - start_time:.2f} seconds.")
    spark.stop()
