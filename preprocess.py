import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import os

# Set environment variables for Spark and Java
os.environ['JAVA_HOME'] = '/opt/bitnami/java'
os.environ['SPARK_HOME'] = '/opt/bitnami/spark'
os.environ['PYARROW_IGNORE_TIMEZONE'] = '1'  # Fix timezone warning

# Initialize Spark Session with increased memory configurations
spark = SparkSession.builder \
    .appName("WeatherPrediction") \
    .config("spark.executor.memory", "64g") \
    .config("spark.executor.cores", "16") \
    .config("spark.driver.memory", "32g") \
    .config("spark.sql.shuffle.partitions", "500") \
    .config("spark.network.timeout", "600s") \
    .getOrCreate()

# Define constants
DATETIME_COL = 'datetime'
HUMIDITY_COL = 'humidity'
PRESSURE_COL = 'pressure'
TEMPERATURE_COL = 'temperature'
WIND_DIRECTION_COL = 'wind_direction'
WIND_SPEED_COL = 'wind_speed'
LATITUDE_COL = 'latitude'
LONGITUDE_COL = 'longitude'
CITY_COL = 'city'
COUNTRY_COL = 'country'
WEATHER_CONDITION_COL = 'weather_condition'

# Load datasets from HDFS
hdfs_base_path = "hdfs://namenode:8020/dataset/Makefile"
weather_conditions_df = spark.read.csv(f"{hdfs_base_path}/weather_description.csv", header=True)
humidity_df = spark.read.csv(f"{hdfs_base_path}/humidity.csv", header=True)
pressure_df = spark.read.csv(f"{hdfs_base_path}/pressure.csv", header=True)
temperature_df = spark.read.csv(f"{hdfs_base_path}/temperature.csv", header=True)
city_attributes_df = spark.read.csv(f"{hdfs_base_path}/city_attributes.csv", header=True)
wind_direction_df = spark.read.csv(f"{hdfs_base_path}/wind_direction.csv", header=True)
wind_speed_df = spark.read.csv(f"{hdfs_base_path}/wind_speed.csv", header=True)

# Convert city_attributes to pandas for iteration
city_attributes_pd = city_attributes_df.toPandas()

def filter_dataframe_by_city_column(dataframe, city_name: str, new_column_name: str):
    """Filter dataframe by city and rename column"""
    return dataframe.withColumn(new_column_name, col(city_name)) \
                   .select([DATETIME_COL, new_column_name])

# Process and combine weather measurements
weather_measurements_df = None

for _, row in city_attributes_pd.iterrows():
    city = row['City']
    country = row['Country']
    latitude = float(row['Latitude'])
    longitude = float(row['Longitude'])

    # Create dataframes list for each measurement type
    dataframes = [
        filter_dataframe_by_city_column(humidity_df, city, HUMIDITY_COL),
        filter_dataframe_by_city_column(pressure_df, city, PRESSURE_COL),
        filter_dataframe_by_city_column(temperature_df, city, TEMPERATURE_COL),
        filter_dataframe_by_city_column(wind_direction_df, city, WIND_DIRECTION_COL),
        filter_dataframe_by_city_column(wind_speed_df, city, WIND_SPEED_COL),
        filter_dataframe_by_city_column(weather_conditions_df, city, WEATHER_CONDITION_COL)
    ]

    # Join dataframes
    joined_df = dataframes[0]
    for df in dataframes[1:]:
        joined_df = joined_df.join(df, [DATETIME_COL])

    # Add city attributes
    joined_df = joined_df.withColumn(CITY_COL, lit(city)) \
                        .withColumn(COUNTRY_COL, lit(country)) \
                        .withColumn(LATITUDE_COL, lit(latitude)) \
                        .withColumn(LONGITUDE_COL, lit(longitude))

    # Union with main dataframe
    weather_measurements_df = weather_measurements_df.union(joined_df) if weather_measurements_df else joined_df

# Persist the DataFrame before converting to pandas to avoid excessive memory use
weather_measurements_df.persist()

# Convert to pandas for preprocessing (take a sample or limit if necessary)
data = weather_measurements_df.limit(10000).toPandas()

# Print data information
print("Data Types of Columns:")
print(data.dtypes)

print("\nSum of Null Values in Each Column:")
print(data.isnull().sum())

# Remove rows with null values
data_cleaned = data.dropna()
print("\nData after removing null values:")
print(data_cleaned.isnull().sum())

# Handle duplicate values
duplicate_count = data_cleaned.duplicated().sum()
print(f"\nSum of Duplicate Rows: {duplicate_count}")
data_cleaned = data_cleaned.drop_duplicates()
print(f"Sum of Duplicate Rows after removal: {data_cleaned.duplicated().sum()}")

# Aggregate weather conditions
weather_conditions_dict = {
    'squall': 'thunderstorm', 'thunderstorm': 'thunderstorm',
    'drizzle': 'rainy', 'rain': 'rainy',
    'sleet': 'snowy', 'snow': 'snowy',
    'cloud': 'cloudy',
    'fog': 'foggy', 'mist': 'foggy', 'haze': 'foggy',
    'clear': 'sunny', 'sun': 'sunny'
}

for old_condition, new_condition in weather_conditions_dict.items():
    data_cleaned[WEATHER_CONDITION_COL] = data_cleaned[WEATHER_CONDITION_COL].str.lower().replace(old_condition, new_condition)

# Convert back to Spark DataFrame
df_spark = spark.createDataFrame(data_cleaned)

# Label encoding for the 'weather_condition' column
indexer = StringIndexer(inputCol=WEATHER_CONDITION_COL, outputCol="weather_condition_index")
df_spark = indexer.fit(df_spark).transform(df_spark)

# Drop the original 'weather_condition' column and use the encoded 'weather_condition_index'
df_spark = df_spark.drop(WEATHER_CONDITION_COL)

# Cast string columns to numeric types (double)
df_spark = df_spark.withColumn(HUMIDITY_COL, col(HUMIDITY_COL).cast('double')) \
                   .withColumn(PRESSURE_COL, col(PRESSURE_COL).cast('double')) \
                   .withColumn(TEMPERATURE_COL, col(TEMPERATURE_COL).cast('double')) \
                   .withColumn(WIND_DIRECTION_COL, col(WIND_DIRECTION_COL).cast('double')) \
                   .withColumn(WIND_SPEED_COL, col(WIND_SPEED_COL).cast('double')) \
                   .withColumn(LATITUDE_COL, col(LATITUDE_COL).cast('double')) \
                   .withColumn(LONGITUDE_COL, col(LONGITUDE_COL).cast('double'))

# Feature engineering
feature_cols = [HUMIDITY_COL, PRESSURE_COL, TEMPERATURE_COL, 
                WIND_DIRECTION_COL, WIND_SPEED_COL, LATITUDE_COL, LONGITUDE_COL]

# Create feature vector
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_spark = assembler.transform(df_spark)

# Split the data
train_data = df_spark.sample(fraction=0.8, seed=42)
test_data = df_spark.subtract(train_data)

# Train Random Forest model
rf = RandomForestClassifier(featuresCol="features", 
                           labelCol="weather_condition_index",  # Use the encoded label column
                           numTrees=100)

# Train the model
rf_model = rf.fit(train_data)

# Make predictions
predictions = rf_model.transform(test_data)

# Evaluate the model
evaluator = MulticlassClassificationEvaluator(labelCol="weather_condition_index", 
                                            predictionCol="prediction",
                                            metricName="accuracy")

accuracy = evaluator.evaluate(predictions)
print(f"\nAccuracy: {accuracy}")

# Save model and preprocessed data to HDFS
model_path = "hdfs://namenode:8020/model/weather_prediction_model"
rf_model.write().overwrite().save(model_path)

preprocessed_data_path = "hdfs://namenode:8020/model/weather_preprocessed_data"
df_spark.write.mode("overwrite").parquet(preprocessed_data_path)

print(f"Model saved to HDFS at {model_path}")
print(f"Preprocessed data saved to HDFS at {preprocessed_data_path}")

# Stop Spark session
spark.stop()
