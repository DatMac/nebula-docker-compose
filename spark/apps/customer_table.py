from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StructType, StructField, StringType, DateType
from faker import Faker
import random
from datetime import date, timedelta

# Define the schema for the Customer data
customer_schema = StructType([
    StructField("cust_id", StringType(), True),
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
    StructField("id_expire_date", DateType(), True)
])

# Function to generate a single fake record
def generate_fake_record(_):
    fake = Faker()
    # Generate data for bccs_customer
    cust_id = fake.uuid4()
    cust_type = random.choice(["Individual", "Corporate"])
    name = fake.name()
    birth_date = fake.date_of_birth(minimum_age=18, maximum_age=90)
    sex = random.choice(["Male", "Female"])
    nationality = fake.country()
    status = random.choice(["Active", "Inactive", "Pending"])
    address = fake.address().replace("\n", ", ")
    description = fake.sentence(nb_words=10)

    # Generate data for bccs_identity
    id_type = random.choice(["Passport", "National ID", "Driver License"])
    id_no = fake.ssn()
    id_issue_place = fake.city()
    id_issue_date = fake.date_between(start_date='-10y', end_date='today')
    id_expire_date = id_issue_date + timedelta(days=random.randint(365, 3650))

    return (
        cust_id, cust_type, name, birth_date, sex, nationality, status,
        address, description, id_type, id_no, id_issue_place,
        id_issue_date, id_expire_date
    )

# Main Spark job logic
if __name__ == "__main__":
    # Initialize Spark Session
    spark = SparkSession.builder \
        .appName("GenerateCustomerData") \
        .getOrCreate()

    # Register the UDF
    generate_record_udf = udf(generate_fake_record, customer_schema)

    # Number of records to generate
    num_records = 1_000_000

    # Create an initial DataFrame with a single column 'id' from 0 to num_records - 1
    # The number of partitions can be adjusted based on your cluster's capacity
    initial_df = spark.range(num_records).repartition(9)

    # Apply the UDF to generate the full dataset
    # The UDF will be executed in a distributed manner across the partitions
    customer_df = initial_df.withColumn("customer_data", generate_record_udf("id")).select("customer_data.*")

    # Show a sample of the generated data
    customer_df.show(5)

    # --- Writing data to files in a distributed manner ---

    # Define output paths
    csv_output_path = "hdfs://namenode:8020/telecom/customer_csv"
    parquet_output_path = "hdfs://namenode:8020/telecom/customer_parquet"

    # Write the DataFrame to CSV format
    # The 'header' option will write the column names as the first line
    print(f"Writing {num_records} records to CSV at: {csv_output_path}")
    customer_df.write.mode("overwrite").option("header", "true").csv(csv_output_path)
    print("Successfully wrote to CSV.")

    # Write the DataFrame to Parquet format
    # Parquet is a columnar format that is highly optimized for Spark
    print(f"Writing {num_records} records to Parquet at: {parquet_output_path}")
    customer_df.write.mode("overwrite").parquet(parquet_output_path)
    print("Successfully wrote to Parquet.")

    # Stop the Spark Session
    spark.stop()
