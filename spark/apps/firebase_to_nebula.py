import os
import json
import re
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

from pyspark.sql import SparkSession
# Added 'posexplode' to the imports here
from pyspark.sql.functions import col, unix_timestamp, to_timestamp, lit, explode, posexplode, when
from pyspark.sql.types import LongType

# --- CONFIGURATION ---
NEBULA_META_ADDRESS = "metad0:9559,metad1:9559,metad2:9559"
NEBULA_GRAPH_ADDRESS = "graphd:9669,graphd1:9669,graphd2:9669"
NEBULA_SPACE = "quiz_recsys"
NEBULA_USER = "root"
NEBULA_PASSWORD = "nebula"
WRITE_BATCH_SIZE = 1024

# --- 1. FIRESTORE READING LOGIC ---

def recursive_clean(obj):
    """
    Recursively converts Firestore types (DateTime, Refs) to JSON-friendly types.
    """
    if isinstance(obj, dict):
        return {k: recursive_clean(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_clean(v) for v in obj]
    elif "datetime" in str(type(obj)) or "DatetimeWithNanoseconds" in str(type(obj)):
        return str(obj)
    elif "DocumentReference" in str(type(obj)):
        return obj.path
    elif "GeoPoint" in str(type(obj)):
        return str(obj)
    else:
        return obj

def get_firestore_collection(collection_name, limit=2000):
    """
    Reads from Firestore and returns a list of JSON strings.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cred_path = os.path.join(script_dir, "serviceAccountKey.json")

    if not firebase_admin._apps:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)

    db = firestore.client()
    print(f"--- [Firestore] Fetching {collection_name} ---")
    docs = db.collection(collection_name).limit(limit).stream()
    
    data = []
    for doc in docs:
        doc_dict = doc.to_dict()
        doc_dict['firestore_id'] = doc.id
        
        clean_dict = recursive_clean(doc_dict)
        
        if collection_name == 'quizResults' and 'answers' in clean_dict:
            clean_dict['answers'] = json.dumps(clean_dict['answers'])
            
        data.append(json.dumps(clean_dict))
    
    return data

def load_df_from_firestore(spark, collection_name):
    json_data = get_firestore_collection(collection_name)
    if not json_data:
        print(f"⚠️ Warning: Collection {collection_name} is empty.")
        # Return empty DF with minimal schema to prevent crash
        return spark.createDataFrame([], spark.read.json(spark.sparkContext.parallelize(["{}"])).schema)
        
    rdd = spark.sparkContext.parallelize(json_data)
    return spark.read.json(rdd)

# --- 2. NEBULA WRITING LOGIC ---

def write_to_nebula(df, nebula_type, name, id_field=None, src_field=None, dst_field=None, rank_field=None):
    if df.rdd.isEmpty():
        print(f"Skipping {name} (No data)")
        return

    print(f"--- Writing {nebula_type.upper()}: {name} ---")
    df.printSchema() # Uncomment for debugging

    writer = df.write.format("com.vesoft.nebula.connector.NebulaDataSource") \
        .option("operateType", "write") \
        .option("metaAddress", NEBULA_META_ADDRESS) \
        .option("graphAddress", NEBULA_GRAPH_ADDRESS) \
        .option("spaceName", NEBULA_SPACE) \
        .option("user", NEBULA_USER) \
        .option("passwd", NEBULA_PASSWORD) \
        .option("type", nebula_type) \
        .option("label", name) \
        .option("writeMode", "insert") \
        .option("batch", WRITE_BATCH_SIZE)

    if nebula_type == "vertex":
        writer.option("vertexField", id_field)
        writer.option("vidPolicy", "")
    else:
        writer.option("srcVertexField", src_field)
        writer.option("dstVertexField", dst_field)
        writer.option("srcPolicy", "")
        writer.option("dstPolicy", "")
        if rank_field:
            writer.option("rankField", rank_field)
        else:
            writer.option("rankField", "")

    writer.save()
    print(f"✅ Successfully wrote {name}")

# --- 3. TRANSFORMATIONS ---

def to_nebula_timestamp(col_name):
    return unix_timestamp(to_timestamp(col(col_name))).cast(LongType())

def main():
    spark = SparkSession.builder \
        .appName("FirestoreToNebula") \
        .master("spark://spark-master:7077") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    # ==========================================
    # A. LOAD RAW DATA
    # ==========================================
    df_users = load_df_from_firestore(spark, "users")
    df_quizzes = load_df_from_firestore(spark, "quizzes")
    df_categories = load_df_from_firestore(spark, "categories")
    df_results = load_df_from_firestore(spark, "quizResults")
    df_reviews = load_df_from_firestore(spark, "quizReviews")

    # ==========================================
    # B. WRITE VERTICES
    # ==========================================
    
    # 1. User
    v_user = df_users.select(
        col("firestore_id").alias("vid"),
        to_nebula_timestamp("createdAt").alias("created_at"),
        col("isActive").alias("is_active")
    ).na.fill({"is_active": True})
    write_to_nebula(v_user, "vertex", "user", id_field="vid")

    # 2. Quiz
    v_quiz = df_quizzes.select(
        col("firestore_id").alias("vid"),
        to_nebula_timestamp("createdAt").alias("created_at"),
        col("difficulty"),
        col("isPublic").alias("is_public")
    )
    write_to_nebula(v_quiz, "vertex", "Quiz", id_field="vid")

    # 3. Category
    v_category = df_categories.select(
        col("firestore_id").alias("vid"),
        col("name")
    )
    write_to_nebula(v_category, "vertex", "Category", id_field="vid")

    # 4. Keyword
    v_keyword = df_quizzes.select(explode("tags").alias("tag_name")) \
        .select(col("tag_name").alias("vid"), col("tag_name").alias("name")) \
        .distinct()
    write_to_nebula(v_keyword, "vertex", "Keyword", id_field="vid")

    # ==========================================
    # C. WRITE EDGES
    # ==========================================

    # 1. TOOK
    e_took = df_results.select(
        col("userId").alias("src"),
        col("quizId").alias("dst"),
        col("score").cast("int"),
        col("timeSpent").cast("int").alias("time_spent"),
        when(col("score") >= 50, True).otherwise(False).alias("is_pass"),
        to_nebula_timestamp("completedAt").alias("created_at")
    ).dropna(subset=["src", "dst"])
    write_to_nebula(e_took, "edge", "TOOK", src_field="src", dst_field="dst")

    # 2. RATED
    e_rated = df_reviews.select(
        col("userId").alias("src"),
        col("quizId").alias("dst"),
        col("rating").cast("int"),
        to_nebula_timestamp("createdAt").alias("created_at")
    ).dropna(subset=["src", "dst"])
    write_to_nebula(e_rated, "edge", "RATED", src_field="src", dst_field="dst")

    # 3. CREATED
    e_created = df_quizzes.select(
        col("createdBy").alias("src"),
        col("firestore_id").alias("dst"),
        to_nebula_timestamp("createdAt").alias("created_at")
    ).dropna(subset=["src", "dst"])
    write_to_nebula(e_created, "edge", "CREATED", src_field="src", dst_field="dst")

    # 4. BELONGS_TO
    e_belongs = df_quizzes.select(
        col("firestore_id").alias("src"),
        col("category").alias("dst")
    ).dropna(subset=["src", "dst"])
    write_to_nebula(e_belongs, "edge", "BELONGS_TO", src_field="src", dst_field="dst")

    # 5. HAS_KEYWORD
    e_has_keyword = df_quizzes.select(
        col("firestore_id").alias("src"),
        explode("tags").alias("dst")
    ).dropna(subset=["src", "dst"])
    write_to_nebula(e_has_keyword, "edge", "HAS_KEYWORD", src_field="src", dst_field="dst")

    # 6. LIKES_CATEGORY (Fixed Logic)
    e_likes = df_users.select(
        col("firestore_id").alias("src"),
        col("stats.favoriteCategories").alias("cats")
    ).select(
        col("src"),
        posexplode("cats").alias("rank_idx", "dst")
    ).select(
        col("src"),
        col("dst"),
        (col("rank_idx") + 1).alias("rank")
    ).dropna(subset=["src", "dst"])

    write_to_nebula(e_likes, "edge", "LIKES_CATEGORY", src_field="src", dst_field="dst", rank_field="rank")

    print("\n🎉 Job Finished Successfully.")
    spark.stop()

if __name__ == "__main__":
    main()
