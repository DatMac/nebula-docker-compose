import re
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, date_format
from pyspark.sql.types import (
    StructType, StructField, StringType, DateType, TimestampType,
    LongType, DoubleType, BooleanType, IntegerType
)

# --- Configuration (No changes here) ---
HDFS_BASE_PATH = "hdfs://namenode:8020/tmp/telecom"
NEBULA_META_ADDRESS = "metad0:9559,metad1:9559,metad2:9559"
NEBULA_GRAPH_ADDRESS = "graphd:9669,graphd1:9669,graphd2:9669"
NEBULA_SPACE = "telecom"
NEBULA_USER = "root"
NEBULA_PASSWORD = "nebula"
WRITE_BATCH_SIZE = 1024

# --- DataFrame Schema Definitions (No changes here) ---
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
makes_call_schema = StructType([
    StructField(":START_ID(Phone_Number)", StringType(), False), StructField(":END_ID(Phone_Number)", StringType(), False),
    StructField(":RANK", IntegerType(), True), StructField("call_timestamp", TimestampType(), True),
    StructField("duration_seconds", IntegerType(), True), StructField("call_type", StringType(), True),
    StructField("call_result", StringType(), True), StructField("start_tower_id", StringType(), True),
    StructField("end_tower_id", StringType(), True), StructField("jitter_ms", DoubleType(), True),
    StructField("packet_loss_percent", DoubleType(), True)
])
sends_sms_schema = StructType([
    StructField(":START_ID(Phone_Number)", StringType(), False), StructField(":END_ID(Phone_Number)", StringType(), False),
    StructField(":RANK", IntegerType(), True), StructField("sms_timestamp", TimestampType(), True),
    StructField("message_length", IntegerType(), True)
])
uses_data_schema = StructType([
    StructField(":START_ID(Device)", StringType(), False), StructField(":END_ID(Cell_Tower)", StringType(), False),
    StructField(":RANK", IntegerType(), True), StructField("session_start_time", TimestampType(), True),
    StructField("session_end_time", TimestampType(), True), StructField("data_uploaded_mb", DoubleType(), True),
    StructField("data_downloaded_mb", DoubleType(), True), StructField("network_type", StringType(), True)
])
uses_app_schema = StructType([
    StructField(":START_ID(Device)", StringType(), False), StructField(":END_ID(Application)", StringType(), False),
    StructField(":RANK", IntegerType(), True), StructField("session_start_time", TimestampType(), True),
    StructField("duration_minutes", IntegerType(), True), StructField("data_consumed_mb", DoubleType(), True)
])


# --- NEW HELPER FUNCTION ---
def format_datetime_columns(df: DataFrame) -> DataFrame:
    """
    Identifies and reformats date and timestamp columns to Nebula's expected string format.
    - TimestampType -> 'yyyy-MM-dd HH:mm:ss'
    - DateType -> 'yyyy-MM-dd'
    """
    for field in df.schema.fields:
        col_name = field.name
        if isinstance(field.dataType, TimestampType):
            print(f"  -> Formatting timestamp column: {col_name}")
            df = df.withColumn(col_name, date_format(col(col_name), 'yyyy-MM-dd HH:mm:ss'))
        elif isinstance(field.dataType, DateType):
            print(f"  -> Formatting date column: {col_name}")
            df = df.withColumn(col_name, date_format(col(col_name), 'yyyy-MM-dd'))
    return df

# --- Clean column names function (No changes) ---
def clean_dataframe_columns(df: DataFrame) -> DataFrame:
    new_columns = []
    for col_name in df.columns:
        if ":START_ID" in col_name:
            new_columns.append("src")
        elif ":END_ID" in col_name:
            new_columns.append("dst")
        elif ":RANK" in col_name:
            new_columns.append("rank")
        else:
            cleaned_name = re.split(r'[:(]', col_name)[0]
            new_columns.append(cleaned_name)
    return df.toDF(*new_columns)

# --- UPDATED WRITE FUNCTIONS ---

def write_vertices_to_nebula(spark: SparkSession, tag_name: str, vertex_id_field: str, schema: StructType):
    print(f"\n--- Loading vertices for tag: '{tag_name}' ---")
    hdfs_path = f"{HDFS_BASE_PATH}/vertices/{tag_name}"
    
    df = spark.read.schema(schema).option("header", "true").csv(hdfs_path)
    
    # *** ADDED THIS STEP ***
    df_formatted = format_datetime_columns(df)
    
    df_clean = clean_dataframe_columns(df_formatted)
    print(f"Read {df_clean.count()} rows from {hdfs_path}. Schema used:")
    df_clean.printSchema()
    
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

def write_edges_to_nebula(spark: SparkSession, edge_name: str, schema: StructType, src_id_field: str = "src", dst_id_field: str = "dst", rank_field: str = ""):
    print(f"\n--- Loading edges for type: '{edge_name}' ---")
    hdfs_path = f"{HDFS_BASE_PATH}/edges/{edge_name}"
    
    df = spark.read.schema(schema).option("header", "true").csv(hdfs_path)
    
    # *** ADDED THIS STEP ***
    df_formatted = format_datetime_columns(df)

    df_clean = clean_dataframe_columns(df_formatted)
    print(f"Read {df_clean.count()} rows from {hdfs_path}. Schema used:")
    df_clean.printSchema()

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
        .option("writeMode", "insert") \
        .option("batch", WRITE_BATCH_SIZE)

    if rank_field:
        writer.option("rankField", rank_field)
    else:
        writer.option("rankField", "")

    writer.save()
        
    print(f"Successfully wrote edges for type '{edge_name}'.")

# --- Main function (No changes) ---
def main():
    spark = SparkSession.builder \
        .appName("HDFS to NebulaGraph Telecom Loader with Schema") \
        .master("spark://spark-master:7077") \
        .getOrCreate()

    print("Spark Session created. Starting data load process...")

    # Load all vertex data using their defined schemas
    write_vertices_to_nebula(spark, "Customer", "customerID", customer_schema)
    write_vertices_to_nebula(spark, "Device", "deviceID", device_schema)
    write_vertices_to_nebula(spark, "Phone_Number", "phoneNumberID", phone_number_schema)
    write_vertices_to_nebula(spark, "Cell_Tower", "towerID", cell_tower_schema)
    write_vertices_to_nebula(spark, "Service_Plan", "planID", service_plan_schema)
    write_vertices_to_nebula(spark, "Application", "appID", application_schema)
    
    # Load all edge data using their defined schemas
    write_edges_to_nebula(spark, "HAS_DEVICE", has_device_schema)
    write_edges_to_nebula(spark, "HAS_PHONE_NUMBER", has_phone_number_schema)
    write_edges_to_nebula(spark, "SUBSCRIBES_TO", subscribes_to_schema)
    write_edges_to_nebula(spark, "MAKES_CALL", makes_call_schema, rank_field="rank")
    write_edges_to_nebula(spark, "SENDS_SMS", sends_sms_schema, rank_field="rank")
    write_edges_to_nebula(spark, "USES_DATA", uses_data_schema, rank_field="rank")
    write_edges_to_nebula(spark, "USES_APP", uses_app_schema, rank_field="rank")

    print("\nData loading job finished successfully!")
    spark.stop()

if __name__ == "__main__":
    main()
