import os
import json
import firebase_admin
from firebase_admin import credentials, firestore
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lit, avg, count, sum as _sum, when, 
    current_timestamp, from_json, max as _max, 
    datediff, current_date, log,
    coalesce, concat_ws, udf
)
from pyspark.sql.types import ArrayType, StringType, FloatType

# --- CONFIGURATION ---
CASSANDRA_HOST = "cassandra"
CASSANDRA_PORT = "9042"
CASSANDRA_DATACENTER = "datacenter1"
CASSANDRA_KEYSPACE = "feature_store"
SPARK_PACKAGES = (
    "com.datastax.spark:spark-cassandra-connector_2.11:2.4.3,"
    "com.twitter:jsr166e:1.1.0"
)

GOOGLE_API_KEY = ""
EMBEDDING_MODEL = "models/text-embedding-004"

# --- FIRESTORE HELPERS (Reused from your example) ---
def recursive_clean(obj):
    if isinstance(obj, dict):
        return {k: recursive_clean(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_clean(v) for v in obj]
    elif "datetime" in str(type(obj)) or "DatetimeWithNanoseconds" in str(type(obj)):
        return str(obj)
    elif "DocumentReference" in str(type(obj)):
        return obj.path
    else:
        return obj

def get_firestore_collection(collection_name, limit=5000):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cred_path = os.path.join(script_dir, "serviceAccountKey.json") # Ensure this exists
    if not firebase_admin._apps:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    docs = db.collection(collection_name).limit(limit).stream()
    data = []
    for doc in docs:
        doc_dict = doc.to_dict()
        doc_dict['firestore_id'] = doc.id
        data.append(json.dumps(recursive_clean(doc_dict)))
    return data

def load_df_from_firestore(spark, collection_name):
    json_data = get_firestore_collection(collection_name)
    if not json_data:
        return spark.createDataFrame([], spark.read.json(spark.sparkContext.parallelize(["{}"])).schema)
    rdd = spark.sparkContext.parallelize(json_data)
    return spark.read.json(rdd)

# --- MAIN FEATURE LOGIC ---

def calculate_user_features(df_users, df_results):
    print("--- Calculating User Features ---")

    # 1. Aggregate Quiz Results per User
    user_stats = df_results.groupBy("userId").agg(
        avg("score").alias("raw_avg_score"),
        count("quizId").alias("total_taken"),
        (_sum(when(col("score") >= 50, 1).otherwise(0)) / count("quizId")).alias("tenacity"),
        _max("completedAt").alias("last_active_raw")
    ).withColumnRenamed("userId", "stats_user_id")

    # 2. Calculate Velocity
    df_results_dates = df_results.withColumn("date_completed", col("completedAt").cast("date"))

    recent_activity = df_results_dates.filter(
        datediff(current_date(), col("date_completed")) <= 7
    ).groupBy("userId").agg(count("quizId").alias("activity_velocity")) \
     .withColumnRenamed("userId", "activity_user_id")

    # 3. Join with Users table using ALIASES
    u = df_users.select("firestore_id", "preferences", "stats").alias("u")
    s = user_stats.alias("s")
    a = recent_activity.alias("a")

    features = u.join(
        s,
        col("u.firestore_id") == col("s.stats_user_id"),
        "left"
    ).join(
        a,
        col("u.firestore_id") == col("a.activity_user_id"),
        "left"
    )

    # 4. Final Formatting (FIXED)
    final_user_features = features.select(
        col("u.firestore_id").alias("user_id"),
        (col("s.raw_avg_score") / 100).cast("float").alias("skill_level"),

        # Handle nulls
        coalesce(col("s.tenacity"), lit(0.0)).cast("float").alias("tenacity"),
        coalesce(col("a.activity_velocity"), lit(0)).cast("int").alias("activity_velocity"),

        # --- REMOVED 'has_medium_stats' TO MATCH CASSANDRA SCHEMA ---

        current_timestamp().alias("last_updated")
    ).withColumn(
        "preferred_difficulty", lit("Medium")
    )

    return final_user_features

def get_embedding_rest(text):
    import requests
    import json
    import time

    # 1. Handle Empty Input immediately
    if not text or len(str(text).strip()) == 0:
        # Return a "Zero Vector" or None. None is safer for now.
        return None
        
    # Correct URL for v1beta
    url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={GOOGLE_API_KEY}"
    
    headers = { "Content-Type": "application/json" }
    
    # 2. Simplified Payload (Model name is in URL, not body)
    payload = {
        "content": {
            "parts": [{ "text": str(text) }]
        }
    }

    try:
        # 3. ADDED verify=False to ignore old SSL certs in Debian Buster
        response = requests.post(url, headers=headers, data=json.dumps(payload), verify=False)
        
        if response.status_code == 200:
            result = response.json()
            return result.get('embedding', {}).get('values', [])
        else:
            # This print goes to the Spark Worker Executor logs (stdout)
            print(f"API FAIL [{response.status_code}]: {response.text}")
            return None
            
    except Exception as e:
        print(f"EXCEPTION: {e}")
        return None

embedding_udf = udf(get_embedding_rest, ArrayType(FloatType()))

def get_embedding_debug(text):
    import requests
    import json
    import time

    if not text:
        return "SKIPPED: Empty Text"
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={GOOGLE_API_KEY}"
    headers = { "Content-Type": "application/json" }
    
    # Simple payload
    payload = {
        "content": { "parts": [{ "text": str(text) }] }
    }

    try:
        # verify=False is CRITICAL for your container
        response = requests.post(url, headers=headers, data=json.dumps(payload), verify=False)
        
        if response.status_code == 200:
            # If successful, just return a success marker
            return "SUCCESS: Got Embedding"
        else:
            # RETURN THE ACTUAL ERROR TO THE CONSOLE
            return f"API ERROR {response.status_code}: {response.text}"
            
    except Exception as e:
        return f"CRITICAL EXCEPTION: {str(e)}"

debug_udf = udf(get_embedding_debug, StringType())

def calculate_quiz_features(df_quizzes, df_results, df_reviews):
    print("--- Calculating Quiz Features ---")

    # 1. Aggregations
    quiz_stats = df_results.groupBy("quizId").agg(
        count("userId").alias("attempts"),
        avg("score").alias("avg_user_score")
    ).withColumn(
        "popularity_score", log(col("attempts") + 1)
    ).withColumn(
        "actual_difficulty", (100 - col("avg_user_score")) / 100
    )

    review_stats = df_reviews.groupBy("quizId").agg(
        avg("rating").alias("quality_rating")
    )

    # 2. Join Metadata (Using aliases to avoid Cartesian Product errors)
    q = df_quizzes.alias("q")
    s = quiz_stats.alias("s")
    r = review_stats.alias("r")

    features = q.join(
        s, col("q.firestore_id") == col("s.quizId"), "left"
    ).join(
        r, col("q.firestore_id") == col("r.quizId"), "left"
    )

    # 3. Prepare Text for Embedding
    features_with_text = features.withColumn(
        "combined_text", 
        concat_ws(" ", 
            col("q.title"), 
            col("q.description"), 
            col("q.category"),
            concat_ws(" ", col("q.tags"))
        )
    )

    # print("--- DEBUG: CHECKING INPUT TEXT ---")
    # features_with_text.select("firestore_id", "combined_text").show(5, truncate=False)
    
    # features_with_debug = features_with_text.withColumn(
    #    "embedding_status", 
    #    debug_udf(col("combined_text"))
    # )

    # print("--- API RESPONSE LOG ---")
    # features_with_debug.select("firestore_id", "embedding_status").show(10, truncate=False)

    # 4. Generate Embeddings (Apply REST UDF)
    print("--- Generating Embeddings via REST API ---")
    features_with_embedding = features_with_text.withColumn(
        "embedding", 
        embedding_udf(col("combined_text"))
    )

    # 5. Final Select
    final_quiz_features = features_with_embedding.select(
        col("q.firestore_id").alias("quiz_id"),
        coalesce(col("s.popularity_score"), lit(0.0)).cast("float").alias("popularity_score"),
        coalesce(col("s.actual_difficulty"), lit(0.5)).cast("float").alias("actual_difficulty"),
        coalesce(col("r.quality_rating"), lit(0.0)).cast("float").alias("quality_rating"),
        col("q.category"),
        col("q.isPublic").cast("boolean").alias("is_public"),
        col("q.tags"),
        col("embedding"),
        current_timestamp().alias("last_updated")
    )

    return final_quiz_features

# --- EXECUTION ---

if __name__ == "__main__":
    SPARK_PACKAGES = (
        "com.datastax.spark:spark-cassandra-connector_2.11:2.4.3,"
        "com.twitter:jsr166e:1.1.0"
    )

    spark = SparkSession.builder \
        .appName("FeatureEngineeringToCassandra") \
        .config("spark.jars.packages", SPARK_PACKAGES) \
        .config("spark.cassandra.connection.host", CASSANDRA_HOST) \
        .config("spark.cassandra.connection.port", CASSANDRA_PORT) \
        .config("spark.cassandra.connection.local_dc", CASSANDRA_DATACENTER) \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    # 1. Load Data
    df_users = load_df_from_firestore(spark, "users")
    df_quizzes = load_df_from_firestore(spark, "quizzes")
    df_results = load_df_from_firestore(spark, "quizResults")
    df_reviews = load_df_from_firestore(spark, "quizReviews")

    # 2. Compute Features
    user_features_df = calculate_user_features(df_users, df_results)
    quiz_features_df = calculate_quiz_features(df_quizzes, df_results, df_reviews)

    # 3. Write to Cassandra
    print("Writing User Features...")
    user_features_df.write \
        .format("org.apache.spark.sql.cassandra") \
        .options(table="user_features", keyspace=CASSANDRA_KEYSPACE) \
        .mode("append") \
        .save()

    print("Writing Quiz Features...")
    quiz_features_df.write \
        .format("org.apache.spark.sql.cassandra") \
        .options(table="quiz_features", keyspace=CASSANDRA_KEYSPACE) \
        .mode("append") \
        .save()

    print("Feature Engineering Complete.")
    spark.stop()
