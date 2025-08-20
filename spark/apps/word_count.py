import sys
from pyspark.sql import SparkSession

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: word_count.py <output_path>")
        sys.exit(-1)

    output_path = sys.argv[1]

    # Create a SparkSession
    spark = SparkSession.builder.appName("AirflowWordCount").getOrCreate()

    # Create an RDD from a list of strings
    # In a real job, you would read from HDFS, S3, etc.
    data = [
        "hello spark from airflow",
        "testing the spark and airflow integration",
        "spark job submitted by airflow",
        "hello world hello spark"
    ]
    rdd = spark.sparkContext.parallelize(data)

    # Perform the word count
    counts = rdd.flatMap(lambda line: line.split(" ")) \
                .map(lambda word: (word, 1)) \
                .reduceByKey(lambda a, b: a + b)

    # Save the result to the specified HDFS path
    counts.saveAsTextFile(output_path)

    print(f"Successfully wrote word count output to {output_path}")

    # Stop the SparkSession
    spark.stop()
