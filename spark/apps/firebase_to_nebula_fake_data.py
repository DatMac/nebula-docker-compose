import os
import json
import re
import random
import uuid
from datetime import datetime, timedelta

# --- Third Party Imports ---
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from faker import Faker

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, unix_timestamp, to_timestamp, lit, explode, posexplode, when
from pyspark.sql.types import LongType

# --- CONFIGURATION ---
NEBULA_META_ADDRESS = "metad0:9559,metad1:9559,metad2:9559"
NEBULA_GRAPH_ADDRESS = "graphd:9669,graphd1:9669,graphd2:9669"
NEBULA_SPACE = "quiz_recsys"
NEBULA_USER = "root"
NEBULA_PASSWORD = "nebula"
WRITE_BATCH_SIZE = 1024

# --- SYNTHETIC DATA CONFIGURATION ---
GENERATE_FAKE_DATA = True
NUM_FAKE_USERS = 1000
NUM_FAKE_CATEGORIES = 20
NUM_FAKE_QUIZZES = 2000
NUM_FAKE_RESULTS = 10000 
NUM_FAKE_REVIEWS = 5000 
PARALLELISM = 100  # High slice count to prevent "Task too large" errors

# --- 1. FIRESTORE READING LOGIC (Returns List of Strings) ---

def recursive_clean(obj):
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

def get_firestore_json_list(collection_name, limit=2000):
    """
    Reads from Firestore and returns a list of JSON strings.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cred_path = os.path.join(script_dir, "serviceAccountKey.json")

    try:
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
    except Exception as e:
        print(f"⚠️ Warning: Could not connect to Firestore or read {collection_name}. Error: {e}")
        return []

# --- 2. SYNTHETIC DATA GENERATION (Returns List of Strings) ---

def generate_synthetic_json_lists():
    fake = Faker()
    print(f"--- 🎲 Generating Synthetic Data (Full Schema) ---")

    # 1. Fake Categories
    fake_category_ids = [f"fake_cat_{uuid.uuid4().hex[:8]}" for _ in range(NUM_FAKE_CATEGORIES)]
    cats_json = []
    for cat_id in fake_category_ids:
        cats_json.append(json.dumps({
            "firestore_id": cat_id,
            "name": fake.word().capitalize(),
            "icon": fake.image_url(),
            "description": fake.sentence()
        }))
    
    # 2. Fake Users (Complex Schema)
    fake_user_ids = [f"fake_user_{uuid.uuid4().hex[:8]}" for _ in range(NUM_FAKE_USERS)]
    users_json = []
    for uid in fake_user_ids:
        fav_cats = random.sample(fake_category_ids, k=random.randint(0, 3))
        
        # Exact schema match for 'stats' to ensure compatibility
        stats = {
            "achievements": [fake.word() for _ in range(random.randint(0, 3))],
            "averageScore": random.randint(40, 100),
            "bestScore": random.randint(80, 100),
            "difficultyStats": {
                "easy": {"attempts": random.randint(1, 20), "bestScore": random.randint(50, 100)},
                "medium": {"attempts": random.randint(1, 20), "bestScore": random.randint(40, 100)},
                "hard": {"attempts": random.randint(0, 10), "bestScore": random.randint(0, 90)}
            },
            "favoriteCategories": fav_cats,
            "lastQuizDate": fake.date_time_between(start_date='-1M', end_date='now').isoformat(),
            "streak": random.randint(0, 30),
            "totalCorrectAnswers": random.randint(10, 500),
            "totalQuestions": random.randint(20, 1000),
            "totalQuizzes": random.randint(5, 100),
            "totalQuizzesCreated": random.randint(0, 5),
            "totalQuizzesTaken": random.randint(5, 100),
            "totalTimeSpent": random.randint(1000, 50000)
        }

        users_json.append(json.dumps({
            "firestore_id": uid,
            "createdAt": fake.date_time_between(start_date='-2y', end_date='now').isoformat(),
            "email": fake.email(),
            "displayName": fake.name(),
            "photoURL": fake.image_url(),
            "isActive": fake.boolean(chance_of_getting_true=90),
            "stats": stats,
            "roles": ["user"],
            "is_fake": True
        }))

    # 3. Fake Quizzes
    fake_quiz_ids = [f"fake_quiz_{uuid.uuid4().hex[:8]}" for _ in range(NUM_FAKE_QUIZZES)]
    quizzes_json = []
    difficulties = ["Easy", "Medium", "Hard"]
    
    for qid in fake_quiz_ids:
        questions = []
        for i in range(random.randint(5, 10)):
            questions.append({
                "id": i,
                "question": fake.sentence() + "?",
                "options": [fake.word() for _ in range(4)],
                "correctAnswer": fake.word()
            })

        quizzes_json.append(json.dumps({
            "firestore_id": qid,
            "title": fake.sentence(nb_words=4),
            "description": fake.paragraph(nb_sentences=2),
            "createdAt": fake.date_time_between(start_date='-1y', end_date='now').isoformat(),
            "difficulty": random.choice(difficulties),
            "isPublic": fake.boolean(chance_of_getting_true=80),
            "createdBy": random.choice(fake_user_ids),
            "category": random.choice(fake_category_ids),
            "tags": fake.words(nb=random.randint(1, 5)),
            "timeLimit": random.choice([300, 600, 900]),
            "questions": questions, 
            "totalPlays": random.randint(0, 500),
            "averageRating": round(random.uniform(3.0, 5.0), 1),
            "is_fake": True
        }))

    # 4. Fake Results
    results_json = []
    for _ in range(NUM_FAKE_RESULTS):
        results_json.append(json.dumps({
            "userId": random.choice(fake_user_ids),
            "quizId": random.choice(fake_quiz_ids),
            "score": random.randint(0, 100),
            "timeSpent": random.randint(30, 600),
            "completedAt": fake.date_time_between(start_date='-1y', end_date='now').isoformat(),
            "answers": json.dumps({"q1": "A"}) 
        }))

    # 5. Fake Reviews
    reviews_json = []
    for _ in range(NUM_FAKE_REVIEWS):
        reviews_json.append(json.dumps({
            "userId": random.choice(fake_user_ids),
            "quizId": random.choice(fake_quiz_ids),
            "rating": random.randint(1, 5),
            "comment": fake.sentence(),
            "createdAt": fake.date_time_between(start_date='-1y', end_date='now').isoformat()
        }))

    return users_json, quizzes_json, cats_json, results_json, reviews_json


# --- 3. NEBULA WRITING LOGIC ---

def write_to_nebula(df, nebula_type, name, id_field=None, src_field=None, dst_field=None, rank_field=None):
    if df is None or len(df.head(1)) == 0:
        print(f"Skipping {name} (No data)")
        return

    print(f"--- Writing {nebula_type.upper()}: {name} ({df.count()} rows) ---")
    
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

    try:
        writer.save()
        print(f"✅ Successfully wrote {name}")
    except Exception as e:
        print(f"❌ Failed to write {name}: {str(e)}")

def to_nebula_timestamp(col_name):
    return unix_timestamp(col(col_name).cast("timestamp")).cast(LongType())

def main():
    spark = SparkSession.builder \
        .appName("FirestoreAndFakeToNebula") \
        .master("spark://spark-master:7077") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    sc = spark.sparkContext

    # ==========================================
    # A. FETCH REAL DATA (As Lists)
    # ==========================================
    real_users = get_firestore_json_list("users")
    real_quizzes = get_firestore_json_list("quizzes")
    real_categories = get_firestore_json_list("categories")
    real_results = get_firestore_json_list("quizResults")
    real_reviews = get_firestore_json_list("quizReviews")

    # ==========================================
    # B. GENERATE & MERGE SYNTHETIC DATA
    # ==========================================
    if GENERATE_FAKE_DATA:
        fake_users, fake_quizzes, fake_cats, fake_results, fake_reviews = generate_synthetic_json_lists()
        
        print(f"--- 🔗 Merging Lists: Real({len(real_users)}) + Fake({len(fake_users)}) ---")
        all_users = real_users + fake_users
        all_quizzes = real_quizzes + fake_quizzes
        all_categories = real_categories + fake_cats
        all_results = real_results + fake_results
        all_reviews = real_reviews + fake_reviews
    else:
        all_users = real_users
        all_quizzes = real_quizzes
        all_categories = real_categories
        all_results = real_results
        all_reviews = real_reviews

    # ==========================================
    # C. CREATE DATAFRAMES (Schema Inference)
    # ==========================================
    # We use PARALLELISM (100) to avoid "Task of very large size"
    # Spark will scan the combined JSON and resolve the schema automatically,
    # handling missing fields like 'stats.difficultyStats.hard' gracefully.

    def create_df(json_list):
        if not json_list:
            return None
        rdd = sc.parallelize(json_list, numSlices=PARALLELISM)
        return spark.read.json(rdd)

    df_users = create_df(all_users)
    df_quizzes = create_df(all_quizzes)
    df_categories = create_df(all_categories)
    df_results = create_df(all_results)
    df_reviews = create_df(all_reviews)

    # ==========================================
    # C. SAVE MERGED DATA TO HDFS (For Cassandra Job)
    # ==========================================
    HDFS_BASE = "hdfs://namenode:8020/tmp"
    
    print("--- 💾 Saving Merged Data to HDFS for Feature Store ---")
    
    # We use 'overwrite' so re-running the job updates the data
    df_users.write.mode("overwrite").parquet(f"{HDFS_BASE}/users")
    df_quizzes.write.mode("overwrite").parquet(f"{HDFS_BASE}/quizzes")
    df_results.write.mode("overwrite").parquet(f"{HDFS_BASE}/results")
    df_reviews.write.mode("overwrite").parquet(f"{HDFS_BASE}/reviews")
    
    # Note: We don't necessarily need categories for features, but good to have
    df_categories.write.mode("overwrite").parquet(f"{HDFS_BASE}/categories")

    # ==========================================
    # D. WRITE VERTICES
    # ==========================================
    
    # 1. User
    if df_users and "firestore_id" in df_users.columns:
        v_user = df_users.select(
            col("firestore_id").alias("vid"),
            to_nebula_timestamp("createdAt").alias("created_at"),
            col("isActive").alias("is_active")
        ).na.fill({"is_active": True})
        write_to_nebula(v_user, "vertex", "user", id_field="vid")

    # 2. Quiz
    if df_quizzes and "firestore_id" in df_quizzes.columns:
        v_quiz = df_quizzes.select(
            col("firestore_id").alias("vid"),
            to_nebula_timestamp("createdAt").alias("created_at"),
            col("difficulty"),
            col("isPublic").alias("is_public")
        )
        write_to_nebula(v_quiz, "vertex", "Quiz", id_field="vid")

    # 3. Category
    if df_categories and "firestore_id" in df_categories.columns:
        v_category = df_categories.select(
            col("firestore_id").alias("vid"),
            col("name")
        )
        write_to_nebula(v_category, "vertex", "Category", id_field="vid")

    # 4. Keyword (From Tags)
    if df_quizzes and "tags" in df_quizzes.columns:
        v_keyword = df_quizzes.select(explode("tags").alias("tag_name")) \
            .select(col("tag_name").alias("vid"), col("tag_name").alias("name")) \
            .distinct()
        write_to_nebula(v_keyword, "vertex", "Keyword", id_field="vid")

    # ==========================================
    # E. WRITE EDGES
    # ==========================================

    # 1. TOOK
    if df_results and "userId" in df_results.columns and "quizId" in df_results.columns:
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
    if df_reviews and "userId" in df_reviews.columns and "quizId" in df_reviews.columns:
        e_rated = df_reviews.select(
            col("userId").alias("src"),
            col("quizId").alias("dst"),
            col("rating").cast("int"),
            to_nebula_timestamp("createdAt").alias("created_at")
        ).dropna(subset=["src", "dst"])
        write_to_nebula(e_rated, "edge", "RATED", src_field="src", dst_field="dst")

    # 3. CREATED
    if df_quizzes and "createdBy" in df_quizzes.columns:
        e_created = df_quizzes.select(
            col("createdBy").alias("src"),
            col("firestore_id").alias("dst"),
            to_nebula_timestamp("createdAt").alias("created_at")
        ).dropna(subset=["src", "dst"])
        write_to_nebula(e_created, "edge", "CREATED", src_field="src", dst_field="dst")

    # 4. BELONGS_TO
    if df_quizzes and "category" in df_quizzes.columns:
        e_belongs = df_quizzes.select(
            col("firestore_id").alias("src"),
            col("category").alias("dst")
        ).dropna(subset=["src", "dst"])
        write_to_nebula(e_belongs, "edge", "BELONGS_TO", src_field="src", dst_field="dst")

    # 5. HAS_KEYWORD
    if df_quizzes and "tags" in df_quizzes.columns:
        e_has_keyword = df_quizzes.select(
            col("firestore_id").alias("src"),
            explode("tags").alias("dst")
        ).dropna(subset=["src", "dst"])
        write_to_nebula(e_has_keyword, "edge", "HAS_KEYWORD", src_field="src", dst_field="dst")

    # 6. LIKES_CATEGORY
    if df_users and "stats" in df_users.columns:
        # We need to safely check if 'favoriteCategories' exists inside 'stats' struct
        # Spark SQL allows accessing nested fields even if they are null
        try:
            e_likes = df_users.select(
                col("firestore_id").alias("src"),
                col("stats.favoriteCategories").alias("cats")
            ).filter(col("cats").isNotNull()) \
             .select(
                col("src"),
                posexplode("cats").alias("rank_idx", "dst")
            ).select(
                col("src"),
                col("dst"),
                (col("rank_idx") + 1).alias("rank")
            ).dropna(subset=["src", "dst"])

            write_to_nebula(e_likes, "edge", "LIKES_CATEGORY", src_field="src", dst_field="dst", rank_field="rank")
        except Exception as e:
            print(f"Skipping LIKES_CATEGORY: {e}")

    print("\n🎉 Job Finished Successfully.")
    spark.stop()

if __name__ == "__main__":
    main()
