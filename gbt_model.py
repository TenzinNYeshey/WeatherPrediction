import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, hour, month, count
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.classification import LogisticRegression  # Changed to faster algorithm
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
import os

# Set environment variables
os.environ['JAVA_HOME'] = '/opt/bitnami/java'
os.environ['SPARK_HOME'] = '/opt/bitnami/spark'
os.environ['PYARROW_IGNORE_TIMEZONE'] = '1'

# Initialize Spark with optimized configs
spark = SparkSession.builder \
    .appName("OptimizedWeatherPrediction") \
    .config("spark.executor.memory", "64g") \
    .config("spark.executor.cores", "16") \
    .config("spark.driver.memory", "32g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.network.timeout", "600s") \
    .getOrCreate()

# Constants
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

# Load datasets
hdfs_base_path = "hdfs://namenode:8020/dataset/Makefile"
weather_conditions_df = spark.read.csv(f"{hdfs_base_path}/weather_description.csv", header=True)
humidity_df = spark.read.csv(f"{hdfs_base_path}/humidity.csv", header=True)
pressure_df = spark.read.csv(f"{hdfs_base_path}/pressure.csv", header=True)
temperature_df = spark.read.csv(f"{hdfs_base_path}/temperature.csv", header=True)
city_attributes_df = spark.read.csv(f"{hdfs_base_path}/city_attributes.csv", header=True)
wind_direction_df = spark.read.csv(f"{hdfs_base_path}/wind_direction.csv", header=True)
wind_speed_df = spark.read.csv(f"{hdfs_base_path}/wind_speed.csv", header=True)

def filter_dataframe_by_city_column(dataframe, city_name: str, new_column_name: str):
    return dataframe.withColumn(new_column_name, col(city_name)) \
                   .select([DATETIME_COL, new_column_name])

# Process data (simplified)
weather_measurements_df = None

# Take only first 10 cities for faster processing
for _, row in city_attributes_df.toPandas().head(10).iterrows():
    city = row['City']
    country = row['Country']
    latitude = float(row['Latitude'])
    longitude = float(row['Longitude'])

    dataframes = [
        filter_dataframe_by_city_column(humidity_df, city, HUMIDITY_COL),
        filter_dataframe_by_city_column(pressure_df, city, PRESSURE_COL),
        filter_dataframe_by_city_column(temperature_df, city, TEMPERATURE_COL),
        filter_dataframe_by_city_column(wind_direction_df, city, WIND_DIRECTION_COL),
        filter_dataframe_by_city_column(wind_speed_df, city, WIND_SPEED_COL),
        filter_dataframe_by_city_column(weather_conditions_df, city, WEATHER_CONDITION_COL)
    ]

    joined_df = dataframes[0]
    for df in dataframes[1:]:
        joined_df = joined_df.join(df, [DATETIME_COL])

    joined_df = joined_df.withColumn(CITY_COL, lit(city)) \
                        .withColumn(COUNTRY_COL, lit(country)) \
                        .withColumn(LATITUDE_COL, lit(latitude)) \
                        .withColumn(LONGITUDE_COL, lit(longitude))

    weather_measurements_df = weather_measurements_df.union(joined_df) if weather_measurements_df else joined_df

# Basic Feature Engineering
df = weather_measurements_df \
    .withColumn("hour", hour(col(DATETIME_COL))) \
    .withColumn("month", month(col(DATETIME_COL)))

# Drop nulls and cast columns
df = df.na.drop()

# Convert numeric columns to double
numeric_cols = [HUMIDITY_COL, PRESSURE_COL, TEMPERATURE_COL, 
                WIND_DIRECTION_COL, WIND_SPEED_COL, 
                "hour", "month"]

for col_name in numeric_cols:
    df = df.withColumn(col_name, col(col_name).cast('double'))

# Print class distribution
print("Class Distribution:")
df.groupBy(WEATHER_CONDITION_COL).agg(count("*").alias("count")).show()

# Create pipeline stages
stages = []

# Label indexing
label_indexer = StringIndexer(inputCol=WEATHER_CONDITION_COL, 
                             outputCol="label", 
                             handleInvalid="keep")
stages.append(label_indexer)

# Feature assembly
assembler = VectorAssembler(inputCols=numeric_cols, 
                           outputCol="assembled_features", 
                           handleInvalid="skip")
stages.append(assembler)

# Scaling
scaler = StandardScaler(inputCol="assembled_features", 
                       outputCol="features", 
                       withStd=True, 
                       withMean=True)
stages.append(scaler)

# Logistic Regression classifier
lr = LogisticRegression(maxIter=20, 
                        regParam=0.3, 
                        elasticNetParam=0.8, 
                        family="multinomial")
stages.append(lr)

# Create and fit pipeline
pipeline = Pipeline(stages=stages)

# Split data (80-20)
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Fit model
print("Training model...")
model = pipeline.fit(train_data)

# Make predictions
predictions = model.transform(test_data)

# Evaluate
evaluator = MulticlassClassificationEvaluator(labelCol="label", 
                                            predictionCol="prediction", 
                                            metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy}")

# Save model and data
model_path = "hdfs://namenode:8020/model/weather_gbt_model"
model.write().overwrite().save(model_path)

preprocessed_data_path = "hdfs://namenode:8020/model/weather_preprocessed_data_1"
predictions.write.mode("overwrite").parquet(preprocessed_data_path)

print(f"Model saved to HDFS at {model_path}")
print(f"Preprocessed data saved to HDFS at {preprocessed_data_path}")
print(f"Accuracy: {accuracy}")
print("Class Distribution:")
spark.stop()


