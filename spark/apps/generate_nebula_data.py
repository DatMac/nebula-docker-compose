import random
import uuid
from datetime import datetime, timedelta

from faker import Faker
from pyspark.sql import SparkSession
from pyspark.sql.types import (DateType, DoubleType, IntegerType, LongType,
                               StringType, StructField, StructType)

# Initialize Faker
fake = Faker()

# --- Configuration ---
NUM_PEOPLE = 10_000
NUM_MOVIES = 50_000
NUM_USERS = 100_000
NUM_RATINGS = 50_000

# HDFS output paths
HDFS_BASE_PATH = "hdfs://namenode:8020/tmp"

# --- Data Generation Functions ---

def create_person_data(num_people):
    """Generates fake data for the 'person' tag."""
    data = []
    for _ in range(num_people):
        data.append((
            f"{uuid.uuid4()}",
            fake.name(),
            fake.date_of_birth(minimum_age=18, maximum_age=80),
            fake.country()
        ))
    return data

def create_movie_data(num_movies):
    """Generates fake data for the 'movie' tag."""
    data = []
    for _ in range(num_movies):
        data.append((
            f"{uuid.uuid4()}",
            ' '.join(fake.words(nb=random.randint(2, 5))).title(),
            random.randint(1980, 2024),
            random.randint(80, 180),
            round(random.uniform(1.0, 10.0), 1),
            fake.sentence(nb_words=20)
        ))
    return data

def create_genre_data():
    """Generates a predefined list of genres."""
    genres = ["Action", "Comedy", "Drama", "Sci-Fi", "Thriller", "Horror", "Romance", "Documentary"]
    return [(f"genre_{g.lower()}", g) for g in genres]

def create_user_data(num_users):
    """Generates fake data for the 'user' tag."""
    data = []
    for _ in range(num_users):
        data.append((
            f"{uuid.uuid4()}",
            fake.user_name(),
            fake.date_between(start_date='-5y', end_date='today'),
            fake.country()
        ))
    return data

def create_relationships(people_ids, movie_ids, user_ids, genre_ids, num_ratings):
    """Generates fake data for all edge types."""
    acted_in_data = []
    directed_data = []
    belongs_to_data = []
    rated_data = []

    # Acted In
    for movie_id in movie_ids:
        actors = random.sample(people_ids, k=random.randint(2, 10))
        for person_id in actors:
            acted_in_data.append((person_id, movie_id, fake.word().capitalize()))

    # Directed
    for movie_id in movie_ids:
        director = random.choice(people_ids)
        directed_data.append((director, movie_id))

    # Belongs To
    for movie_id in movie_ids:
        genres = random.sample(genre_ids, k=random.randint(1, 3))
        for genre_id in genres:
            belongs_to_data.append((movie_id, genre_id))

    # Rated
    for _ in range(num_ratings):
        user_id = random.choice(user_ids)
        movie_id = random.choice(movie_ids)
        rated_data.append((
            user_id,
            movie_id,
            random.randint(1, 5),
            int(datetime.now().timestamp()) - random.randint(0, 31536000) # a rating within the last year
        ))

    return acted_in_data, directed_data, belongs_to_data, rated_data


if __name__ == "__main__":
    # 1. Initialize Spark Session
    spark = SparkSession.builder.appName("NebulaDataGenerator").getOrCreate()

    # 2. Generate Vertex Data
    print("Generating vertex data...")
    people_data = create_person_data(NUM_PEOPLE)
    movie_data = create_movie_data(NUM_MOVIES)
    genre_data = create_genre_data()
    user_data = create_user_data(NUM_USERS)

    # Extract IDs for relationship generation
    people_ids = [p[0] for p in people_data]
    movie_ids = [m[0] for m in movie_data]
    genre_ids = [g[0] for g in genre_data]
    user_ids = [u[0] for u in user_data]

    # 3. Generate Edge Data
    print("Generating edge data...")
    acted_in_data, directed_data, belongs_to_data, rated_data = create_relationships(
        people_ids, movie_ids, user_ids, genre_ids, NUM_RATINGS
    )

    # 4. Create DataFrames
    print("Creating Spark DataFrames...")
    # Vertex DataFrames
    person_df = spark.createDataFrame(people_data, ["personID:ID(person)", "name:string", "birthdate:date", "nationality:string"])
    movie_df = spark.createDataFrame(movie_data, ["movieID:ID(movie)", "title:string", "release_year:int", "runtime_in_min:int", "imdb_rating:double", "plot_summary:string"])
    genre_df = spark.createDataFrame(genre_data, ["genreID:ID(genre)", "name:string"])
    user_df = spark.createDataFrame(user_data, ["userID:ID(user)", "username:string", "join_date:date", "country:string"])

    # Edge DataFrames
    acted_in_df = spark.createDataFrame(acted_in_data, ["src:START_ID(person)", "dst:END_ID(movie)", "role:string"])
    directed_df = spark.createDataFrame(directed_data, ["src:START_ID(person)", "dst:END_ID(movie)"])
    belongs_to_df = spark.createDataFrame(belongs_to_data, ["src:START_ID(movie)", "dst:END_ID(genre)"])
    rated_df = spark.createDataFrame(rated_data, ["src:START_ID(user)", "dst:END_ID(movie)", "rating:int", "timestamp:long"])

    # 5. Write DataFrames to HDFS as CSV
    print(f"Writing data to HDFS at {HDFS_BASE_PATH}...")
    # Write Vertices
    person_df.write.mode("overwrite").option("header", "true").csv(f"{HDFS_BASE_PATH}/vertices/person")
    movie_df.write.mode("overwrite").option("header", "true").csv(f"{HDFS_BASE_PATH}/vertices/movie")
    genre_df.write.mode("overwrite").option("header", "true").csv(f"{HDFS_BASE_PATH}/vertices/genre")
    user_df.write.mode("overwrite").option("header", "true").csv(f"{HDFS_BASE_PATH}/vertices/user")

    # Write Edges
    acted_in_df.write.mode("overwrite").option("header", "true").csv(f"{HDFS_BASE_PATH}/edges/acted_in")
    directed_df.write.mode("overwrite").option("header", "true").csv(f"{HDFS_BASE_PATH}/edges/directed")
    belongs_to_df.write.mode("overwrite").option("header", "true").csv(f"{HDFS_BASE_PATH}/edges/belongs_to")
    rated_df.write.mode("overwrite").option("header", "true").csv(f"{HDFS_BASE_PATH}/edges/rated")

    print("Data generation complete and saved to HDFS.")

    # Stop the Spark Session
    spark.stop()
