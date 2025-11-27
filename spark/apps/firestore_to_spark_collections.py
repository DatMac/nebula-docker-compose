import os
import json
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from pyspark.sql import SparkSession

# ---------------------------------------------------------
# 1. Recursive Cleaning Function
# ---------------------------------------------------------
def recursive_clean(obj):
    """
    Traverses nested dictionaries and lists to convert 
    Firestore-specific types (DateTime, Ref, GeoPoint) into 
    Python native types (String, Float, etc.) compatible with JSON.
    """
    # Handle Dictionary
    if isinstance(obj, dict):
        return {k: recursive_clean(v) for k, v in obj.items()}
    
    # Handle List
    elif isinstance(obj, list):
        return [recursive_clean(v) for v in obj]
    
    # Handle Datetime (Deep check)
    # Checks for standard python datetime OR the Google specific one
    elif "datetime" in str(type(obj)) or "DatetimeWithNanoseconds" in str(type(obj)):
        return str(obj)
    
    # Handle Document References
    elif "DocumentReference" in str(type(obj)):
        return obj.path
        
    # Handle GeoPoints
    elif "GeoPoint" in str(type(obj)):
        return str(obj)
        
    # Return primitive types (str, int, float, bool, None) as is
    else:
        return obj

# ---------------------------------------------------------
# 2. Fetch and Process Data
# ---------------------------------------------------------
def get_firestore_data(collection_name, limit=1000):
    # Initialize App
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cred_path = os.path.join(script_dir, "serviceAccountKey.json")

    if not firebase_admin._apps:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)

    db = firestore.client()

    print(f"\n[Firestore] Fetching from: '{collection_name}' (Limit: {limit})...")
    docs_stream = db.collection(collection_name).limit(limit).stream()

    json_rows = []
    
    for doc in docs_stream:
        doc_data = doc.to_dict()
        doc_data['firestore_id'] = doc.id
        
        # 1. Deep Clean the data (fix nested timestamps)
        clean_doc = recursive_clean(doc_data)

        # 2. HANDLE CONFLICTS (The "Merge Type" Fix)
        # If a field like 'answers' is sometimes a List and sometimes a Map,
        # Spark will crash. We force these known complex fields to be JSON Strings.
        if collection_name == 'quizResults' and 'answers' in clean_doc:
            clean_doc['answers'] = json.dumps(clean_doc['answers'])

        # 3. Convert entire row to JSON String
        # This prepares it for spark.read.json, which handles Nulls/Schemas better
        json_rows.append(json.dumps(clean_doc))

    return json_rows

def main():
    spark = SparkSession.builder \
        .appName("FirebaseDataIngestionFixed") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    target_collections = [
        "users", 
        "user_stats", 
        "quizzes", 
        "quizReviews", 
        "quizResults", 
        "categories"
    ]

    for col in target_collections:
        print(f"==========================================")
        print(f"PROCESSING COLLECTION: {col}")
        print(f"==========================================")
        
        try:
            # Get list of JSON strings
            json_str_list = get_firestore_data(col)
            
            if not json_str_list:
                print(f"⚠️  Collection '{col}' is empty.")
                continue

            print(f"✅ Fetched {len(json_str_list)} records. Inferring Schema...")

            # ---------------------------------------------------------
            # 3. The Robust Spark Read Strategy
            # ---------------------------------------------------------
            # Instead of createDataFrame(list), we parallelize the strings
            # and let Spark's JSON reader handle the schema inference.
            rdd = spark.sparkContext.parallelize(json_str_list)
            df = spark.read.json(rdd)

            print(f"--- Schema for {col} ---")
            df.printSchema()

            print(f"--- Sample Data for {col} ---")
            df.show(5, truncate=True)

        except Exception as e:
            # Print full error stack trace to help debugging
            import traceback
            traceback.print_exc()
            print(f"❌ Error processing '{col}': {e}")

    spark.stop()

if __name__ == "__main__":
    main()
