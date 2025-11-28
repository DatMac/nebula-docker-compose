import os
import random
import requests
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lit, avg, count, sum as _sum, when, 
    current_timestamp, max as _max, 
    datediff, current_date, log,
    coalesce, concat_ws, udf
)
from pyspark.sql.types import ArrayType, FloatType

# --- CONFIGURATION ---
CASSANDRA_HOST = "cassandra"
CASSANDRA_PORT = "9042"
CASSANDRA_DATACENTER = "datacenter1"
CASSANDRA_KEYSPACE = "feature_store"
HDFS_BASE = "hdfs://namenode:8020/tmp"

# Google Gemini Config
GOOGLE_API_KEY = ""
EMBEDDING_DIMENSION = 768

# --- EMBEDDING LOGIC ---
def get_embedding_api(text):
    if not text: return None
    url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={GOOGLE_API_KEY}"
    headers = { "Content-Type": "application/json" }
    payload = { "content": { "parts": [{ "text": str(text) }] } }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), verify=False)
        if response.status_code == 200:
            return response.json().get('embedding', {}).get('values', [])
    except:
        pass
    return None

def get_embedding_random(text):
    # Deterministic random based on text length to keep it slightly consistent, or just pure random
    return [random.uniform(-0.1, 0.1) for _ in range(EMBEDDING_DIMENSION)]

api_embedding_udf = udf(get_embedding_api, ArrayType(FloatType()))
random_embedding_udf = udf(get_embedding_random, ArrayType(FloatType()))

# --- FEATURE ENGINEERING ---

def calculate_user_features(df_users, df_results):
    print("--- Calculating User Features ---")
    
    # Ensure timestamps are correct types (Parquet usually preserves this, but safety first)
    df_results = df_results.withColumn("completedAt", col("completedAt").cast("timestamp"))

    user_stats = df_results.groupBy("userId").agg(
        avg("score").alias("raw_avg_score"),
        count("quizId").alias("total_taken"),
        (_sum(when(col("score") >= 50, 1).otherwise(0)) / count("quizId")).alias("tenacity"),
        _max("completedAt").alias("last_active_raw")
    ).withColumnRenamed("userId", "stats_user_id")

    recent_activity = df_results.filter(
        datediff(current_date(), col("completedAt")) <= 7
    ).groupBy("userId").agg(count("quizId").alias("activity_velocity")) \
     .withColumnRenamed("userId", "activity_user_id")

    u = df_users.alias("u")
    s = user_stats.alias("s")
    a = recent_activity.alias("a")

    features = u.join(s, col("u.firestore_id") == col("s.stats_user_id"), "left") \
                .join(a, col("u.firestore_id") == col("a.activity_user_id"), "left")

    final_user_features = features.select(
        col("u.firestore_id").alias("user_id"),
        (coalesce(col("s.raw_avg_score"), lit(0)) / 100).cast("float").alias("skill_level"),
        coalesce(col("s.tenacity"), lit(0.0)).cast("float").alias("tenacity"),
        coalesce(col("a.activity_velocity"), lit(0)).cast("int").alias("activity_velocity"),
        current_timestamp().alias("last_updated")
    ).withColumn("preferred_difficulty", lit("Medium"))

    return final_user_features

def calculate_quiz_features(df_quizzes, df_results, df_reviews):
    print("--- Calculating Quiz Features ---")

    # 1. Aggregations & RENAME quizId to avoid ambiguity later
    quiz_stats = df_results.groupBy("quizId").agg(
        count("userId").alias("attempts"),
        avg("score").alias("avg_user_score")
    ).withColumn("popularity_score", log(col("attempts") + 1)) \
     .withColumn("actual_difficulty", (100 - col("avg_user_score")) / 100) \
     .withColumnRenamed("quizId", "stats_quiz_id")  # <--- FIX 1: Rename

    review_stats = df_reviews.groupBy("quizId").agg(
        avg("rating").alias("quality_rating")
    ).withColumnRenamed("quizId", "review_quiz_id") # <--- FIX 2: Rename

    # 2. Join
    q = df_quizzes.alias("q")
    s = quiz_stats.alias("s")
    r = review_stats.alias("r")

    # Join using the new unique column names
    features = q.join(s, col("q.firestore_id") == col("s.stats_quiz_id"), "left") \
                .join(r, col("q.firestore_id") == col("r.review_quiz_id"), "left")

    # 3. Prepare Text
    features_with_text = features.withColumn(
        "combined_text", 
        concat_ws(" ", col("q.title"), col("q.description"), col("q.category"), concat_ws(" ", col("q.tags")))
    )

    print("--- Generating Embeddings ---")

    # 4. Handle is_fake flag safely
    # Check if column exists (handling pure real data case)
    if "is_fake" not in features_with_text.columns:
        features_with_text = features_with_text.withColumn("is_fake", lit(False))
    
    # Fill nulls in 'is_fake' (Real data from union usually has nulls here) with False
    features_with_text = features_with_text.fillna({"is_fake": False})

    # 5. Generate Embeddings (Conditional)
    features_with_embedding = features_with_text.withColumn(
        "embedding", 
        when(col("is_fake") == True, random_embedding_udf(col("combined_text")))
        .otherwise(api_embedding_udf(col("combined_text")))
    )

    # 6. Final Select
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

# --- MAIN ---

if __name__ == "__main__":
    SPARK_PACKAGES = (
        "com.datastax.spark:spark-cassandra-connector_2.11:2.4.3,"
        "com.twitter:jsr166e:1.1.0"
    )

    spark = SparkSession.builder \
        .appName("HDFS_to_Cassandra") \
        .config("spark.jars.packages", SPARK_PACKAGES) \
        .config("spark.cassandra.connection.host", CASSANDRA_HOST) \
        .config("spark.cassandra.connection.port", CASSANDRA_PORT) \
        .config("spark.cassandra.connection.local_dc", CASSANDRA_DATACENTER) \
        .master("spark://spark-master:7077") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    print(f"--- 📥 Reading Merged Data from HDFS: {HDFS_BASE} ---")
    
    # 1. Read Parquet from HDFS (This contains Real + Fake data from Job 1)
    try:
        df_users = spark.read.parquet(f"{HDFS_BASE}/users")
        df_quizzes = spark.read.parquet(f"{HDFS_BASE}/quizzes")
        df_results = spark.read.parquet(f"{HDFS_BASE}/results")
        df_reviews = spark.read.parquet(f"{HDFS_BASE}/reviews")
        print(f"Loaded {df_users.count()} Users and {df_quizzes.count()} Quizzes.")
    except Exception as e:
        print(f"❌ Error reading from HDFS. Did you run the Nebula Job first? \n{e}")
        spark.stop()
        exit(1)

    # 2. Compute Features
    user_features_df = calculate_user_features(df_users, df_results)
    quiz_features_df = calculate_quiz_features(df_quizzes, df_results, df_reviews)

    # 3. Write to Cassandra
    print("Writing User Features to Cassandra...")
    user_features_df.write \
        .format("org.apache.spark.sql.cassandra") \
        .options(table="user_features", keyspace=CASSANDRA_KEYSPACE) \
        .mode("append") \
        .save()

    print("Writing Quiz Features to Cassandra...")
    quiz_features_df.write \
        .format("org.apache.spark.sql.cassandra") \
        .options(table="quiz_features", keyspace=CASSANDRA_KEYSPACE) \
        .mode("append") \
        .save()

    print("✅ Job Complete.")
    spark.stop()
