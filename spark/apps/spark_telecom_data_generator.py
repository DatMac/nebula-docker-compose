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
NUM_CUSTOMERS = 100_000
NUM_CELL_TOWERS = 5000
NUM_CALLS = 2_000_000
NUM_SMS = 5_000_000
NUM_DATA_SESSIONS = 10_000_000
NUM_APP_SESSIONS = 20_000_000

# HDFS output paths
HDFS_BASE_PATH = "hdfs://namenode:8020/tmp/telecom"

# Initialize Faker for data generation
fake = Faker()

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

# --- Helper Functions and Data Generation ---
def get_random_timestamp(start_date="-2y"):
    return fake.date_time_between(start_date=start_date, end_date='now')

def print_progress(current, total, task_name=""):
    if total == 0: return
    update_frequency = max(1, total // 100)
    if current % update_frequency == 0 or current == total:
        percent = 100 * (current / float(total))
        bar_length = 40
        filled_length = int(bar_length * current // total)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        sys.stdout.write(f"\r  -> Progress ({task_name}): |{bar}| {percent:.1f}% Complete ({current}/{total})")
        sys.stdout.flush()
        if current == total: sys.stdout.write("\n")

def create_customer_data(num_customers):
    print(f"--- Generating {num_customers} Customers ---")
    data = []
    for i in range(num_customers):
        data.append((
            str(uuid.uuid4()), f"CUST-{i+1:08d}", fake.first_name(), fake.last_name(), fake.date_of_birth(minimum_age=18, maximum_age=90),
            random.choice(['Male', 'Female', 'Other']), fake.street_address(), fake.city(), fake.state_abbr(), fake.zipcode(),
            fake.email(), fake.phone_number(), random.choice(['Prepaid', 'Postpaid', 'Business']),
            random.choice(['Active', 'Suspended', 'Deactivated']), fake.date_time_between(start_date="-5y", end_date='now'),
            random.randint(300, 850), round(random.uniform(20.0, 150.0), 2), round(random.random(), 3), random.choice([True, False])
        ))
        print_progress(i + 1, num_customers, "Customers")
    return data

def create_devices_and_numbers_for_customers(customer_ids):
    print("--- Generating Devices and Phone Numbers for Customers ---")
    device_data, phone_number_data, has_device_edges, has_phone_number_edges = [], [], [], []
    device_models = ['iPhone 15', 'Samsung Galaxy S25', 'Google Pixel 9', 'iPhone 14 Pro']
    manufacturers = ['Apple', 'Samsung', 'Google']
    num_customers = len(customer_ids)
    for i, cid in enumerate(customer_ids):
        num_assets = random.randint(1, 2)
        for j in range(num_assets):
            is_primary = (j == 0)
            device_vid = str(uuid.uuid4())
            device_data.append((
                device_vid, f"{random.randint(10**14, 10**15 - 1)}", random.choice(device_models), random.choice(manufacturers),
                random.choice(['iOS', 'Android']), f"{random.randint(10, 17)}.{random.randint(0, 5)}",
                fake.date_object(end_datetime=datetime.now()), get_random_timestamp("-1y"), random.choice([True, False]), 'Active'
            ))
            has_device_edges.append((cid, device_vid, get_random_timestamp("-3y"), None, is_primary))
            number_vid = str(uuid.uuid4())
            phone_number_data.append((number_vid, fake.phone_number(), random.randint(1, 200), 'Mobile', get_random_timestamp("-4y"), None))
            has_phone_number_edges.append((cid, number_vid, get_random_timestamp("-4y")))
        print_progress(i + 1, num_customers, "Customer Assets")
    return device_data, phone_number_data, has_device_edges, has_phone_number_edges

def create_cell_tower_data(num_towers):
    print(f"--- Generating {num_towers} Cell Towers ---")
    data = []
    for i in range(num_towers):
        capacity = random.choice([1000, 2000, 5000])
        data.append((
            str(uuid.uuid4()), f"TOWER-{i:04d}", float(fake.latitude()), float(fake.longitude()), fake.city(), fake.state_abbr(),
            fake.zipcode(), random.choice(['4G', '5G']), capacity, random.randint(100, capacity - 50),
            random.choice(['Active', 'Under Maintenance'])
        ))
        print_progress(i + 1, num_towers, "Cell Towers")
    return data

def create_static_vertex_data(catalog_data):
    return [(str(uuid.uuid4()),) + tuple(item.values()) for item in catalog_data]

def create_relationships(customer_ids, phone_number_ids, device_ids, service_plan_ids, cell_tower_ids, app_ids):
    print("--- Generating Relationship (Edge) Data ---")
    subscribes_to_data, makes_call_data, sends_sms_data, uses_data_data, uses_app_data = [], [], [], [], []
    total_customers = len(customer_ids)
    for i, cid in enumerate(customer_ids):
        num_subscriptions = random.randint(1, 3)
        if num_subscriptions > len(service_plan_ids): num_subscriptions = len(service_plan_ids)
        subscribed_plans = random.sample(service_plan_ids, k=num_subscriptions)
        for plan_id in subscribed_plans:
            subscribes_to_data.append((cid, plan_id, get_random_timestamp(), fake.date_object()))
        print_progress(i + 1, total_customers, "SUBSCRIBES_TO")
    for i in range(NUM_CALLS):
        src_num, dst_num = random.sample(phone_number_ids, 2)
        makes_call_data.append((src_num, dst_num, 0, get_random_timestamp(), random.randint(5, 3600), 'Voice', random.choice(['Answered', 'Missed', 'Voicemail']), random.choice(cell_tower_ids), random.choice(cell_tower_ids), round(random.uniform(1.0, 50.0), 2), round(random.uniform(0.0, 2.0), 3)))
        print_progress(i + 1, NUM_CALLS, "MAKES_CALL")
    for i in range(NUM_SMS):
        src_num, dst_num = random.sample(phone_number_ids, 2)
        sends_sms_data.append((src_num, dst_num, 0, get_random_timestamp(), random.randint(5, 160)))
        print_progress(i + 1, NUM_SMS, "SENDS_SMS")
    for i in range(NUM_DATA_SESSIONS):
        start_time = get_random_timestamp()
        uses_data_data.append((random.choice(device_ids), random.choice(cell_tower_ids), 0, start_time, start_time + timedelta(minutes=random.randint(1, 60)), round(random.uniform(0.1, 50.0), 3), round(random.uniform(1.0, 1000.0), 3), random.choice(['4G', '5G'])))
        print_progress(i + 1, NUM_DATA_SESSIONS, "USES_DATA")
    for i in range(NUM_APP_SESSIONS):
        uses_app_data.append((random.choice(device_ids), random.choice(app_ids), 0, get_random_timestamp(), random.randint(1, 120), round(random.uniform(1, 500), 2)))
        print_progress(i + 1, NUM_APP_SESSIONS, "USES_APP")
    return subscribes_to_data, makes_call_data, sends_sms_data, uses_data_data, uses_app_data


if __name__ == "__main__":
    start_time = time.time()
    spark = SparkSession.builder.appName("TelecomDataGenerator").getOrCreate()
    print(f"Spark Session created. Starting data generation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Generate Data in memory
    customer_data = create_customer_data(NUM_CUSTOMERS)
    cell_tower_data = create_cell_tower_data(NUM_CELL_TOWERS)
    service_plan_data = create_static_vertex_data(SERVICE_PLANS)
    application_data = create_static_vertex_data(APPLICATIONS)
    customer_ids = [c[0] for c in customer_data]
    device_data, phone_number_data, has_device_edges, has_phone_number_edges = create_devices_and_numbers_for_customers(customer_ids)
    
    print("\nExtracting vertex IDs for relationship generation...")
    device_ids, phone_number_ids = [d[0] for d in device_data], [p[0] for p in phone_number_data]
    cell_tower_ids, service_plan_ids = [ct[0] for ct in cell_tower_data], [sp[0] for sp in service_plan_data]
    application_ids = [a[0] for a in application_data]
    
    subscribes_to_data, makes_call_data, sends_sms_data, uses_data_data, uses_app_data = create_relationships(
        customer_ids, phone_number_ids, device_ids, service_plan_ids, cell_tower_ids, application_ids
    )

    # 2. DEFINE EXPLICIT SCHEMAS
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
    
    # This schema fixes the error by telling Spark the `deactivation_date` is a nullable TimestampType
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
    
    # Schemas for Edges
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
    
    # 3. Create Spark DataFrames using the explicit schemas
    print("\n--- Creating Spark DataFrames from generated data ---")
    customer_df = spark.createDataFrame(customer_data, schema=customer_schema)
    device_df = spark.createDataFrame(device_data, schema=device_schema)
    phone_number_df = spark.createDataFrame(phone_number_data, schema=phone_number_schema)
    cell_tower_df = spark.createDataFrame(cell_tower_data, schema=cell_tower_schema)
    service_plan_df = spark.createDataFrame(service_plan_data, schema=service_plan_schema)
    application_df = spark.createDataFrame(application_data, schema=application_schema)

    has_device_df = spark.createDataFrame(has_device_edges, schema=has_device_schema)
    has_phone_number_df = spark.createDataFrame(has_phone_number_edges, schema=has_phone_number_schema)
    subscribes_to_df = spark.createDataFrame(subscribes_to_data, schema=subscribes_to_schema)
    
    # For these, inference is less risky, but being explicit is still better.
    # Note that RANK is not part of the nGQL schema but is useful for some loaders. Let's make it an Integer.
    makes_call_df = spark.createDataFrame(makes_call_data, [":START_ID(Phone_Number)", ":END_ID(Phone_Number)", ":RANK", "call_timestamp", "duration_seconds", "call_type", "call_result", "start_tower_id", "end_tower_id", "jitter_ms", "packet_loss_percent"])
    sends_sms_df = spark.createDataFrame(sends_sms_data, [":START_ID(Phone_Number)", ":END_ID(Phone_Number)", ":RANK", "sms_timestamp", "message_length"])
    uses_data_df = spark.createDataFrame(uses_data_data, [":START_ID(Device)", ":END_ID(Cell_Tower)", ":RANK", "session_start_time", "session_end_time", "data_uploaded_mb", "data_downloaded_mb", "network_type"])
    uses_app_df = spark.createDataFrame(uses_app_data, [":START_ID(Device)", ":END_ID(Application)", ":RANK", "session_start_time", "duration_minutes", "data_consumed_mb"])

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
