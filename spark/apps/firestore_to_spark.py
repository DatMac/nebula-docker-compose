import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, MapType

def get_firestore_data(collection_name, limit=1000):
    """
    Connects to Firestore and fetches data as a list of dictionaries.
    This runs on the Driver node.
    """
    # 1. Setup Credentials
    # We look for the key in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cred_path = os.path.join(script_dir, "serviceAccountKey.json")

    # Check if app is already initialized to prevent errors in Spark environment
    if not firebase_admin._apps:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)

    db = firestore.client()

    # 2. Query Firestore
    print(f"--- Fetching data from collection: {collection_name} ---")
    docs_stream = db.collection(collection_name).limit(limit).stream()

    data_list = []
    for doc in docs_stream:
        doc_data = doc.to_dict()
        # Add the document ID to the data so we don't lose it
        doc_data['firestore_id'] = doc.id
        
        # CLEANUP: Spark 2.4 struggles with Firestore Timestamp objects.
        # Convert them to strings.
        for key, value in doc_data.items():
            if str(type(value)) == "<class 'google.api_core.datetime_helpers.DatetimeWithNanoseconds'>":
                doc_data[key] = str(value)
            elif str(type(value)) == "<class 'datetime.datetime'>":
                doc_data[key] = str(value)

        data_list.append(doc_data)

    return data_list

def main():
    # 3. Initialize Spark Session
    spark = SparkSession.builder \
        .appName("FirestoreToSpark") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    # Name of the collection you want to read
    COLLECTION_NAME = "users"  # <--- CHANGE THIS TO YOUR COLLECTION NAME

    # 4. Fetch Data (On Driver)
    try:
        raw_data = get_firestore_data(COLLECTION_NAME)
        
        if not raw_data:
            print("No data found in Firestore.")
            return

        print(f"Successfully fetched {len(raw_data)} records.")

        # 5. Convert to Spark DataFrame
        # We allow Spark to infer schema, or you can define StructType manually for better performance
        df = spark.createDataFrame(raw_data)

        print("--- Spark Schema Inference ---")
        df.printSchema()

        print("--- Sample Data ---")
        df.show(5, truncate=False)

        # Example: Perform a simple transformation
        # df.select("firestore_id", "name").show()

    except Exception as e:
        print(f"Error occurred: {e}")

    spark.stop()

if __name__ == "__main__":
    main()
