from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("Read Parquet").getOrCreate()

# Read Parquet file
# Correct way to specify local path on Windows
df = spark.read.parquet("file:///C:/Users/LAB3/Desktop/WP/Makefile/output/weather_preprocessed_data/*.parquet")


# Show data
df.show()
