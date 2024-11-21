import pandas as pd
import numpy as np
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, lit, hour, month
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler, OneHotEncoder, IndexToString
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from typing import List
import os

# Set environment variables
os.environ['JAVA_HOME'] = '/opt/bitnami/java'
os.environ['SPARK_HOME'] = '/opt/bitnami/spark'
os.environ['PYARROW_IGNORE_TIMEZONE'] = '1'

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("WeatherPrediction") \
    .config("spark.executor.memory", "128g") \
    .config("spark.executor.cores", "32") \
    .config("spark.driver.memory", "64g") \
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
LABEL_COL = "label"
PREDICTION_COL = "prediction"
PREDICTED_TARGET_VARIABLE_COL = "predicted_weather"
FEATURES_COL = "features"

def encoding_pipeline(dataframe: DataFrame,
                     numerical_features: List[str],
                     categorical_features: List[str],
                     target_variable: str,
                     with_std: bool = True,
                     with_mean: bool = False) -> Pipeline:
    """Creates encoding pipeline for features"""
    stages = []
    
    # Target encoding
    label_indexer = StringIndexer(inputCol=target_variable, 
                                 outputCol=LABEL_COL,
                                 handleInvalid="keep")
    stages.append(label_indexer)
    
    # Categorical features encoding
    for categorical_feature in categorical_features:
        string_indexer = StringIndexer(inputCol=categorical_feature, 
                                     outputCol=f"{categorical_feature}_index",
                                     handleInvalid="keep")
        encoder = OneHotEncoder(inputCol=f"{categorical_feature}_index",
                              outputCol=f"{categorical_feature}_encoded")
        stages.extend([string_indexer, encoder])
    
    # Combine all features
    assembler_inputs = numerical_features + \
                      [f"{f}_encoded" for f in categorical_features]
    
    vector_assembler = VectorAssembler(inputCols=assembler_inputs,
                                     outputCol="assembled_features",
                                     handleInvalid="keep")
    stages.append(vector_assembler)
    
    # Scale features
    scaler = StandardScaler(inputCol="assembled_features",
                           outputCol=FEATURES_COL,
                           withStd=with_std,
                           withMean=with_mean)
    stages.append(scaler)
    
    return Pipeline(stages=stages)

def random_forest_pipeline(dataframe: DataFrame,
                         numerical_features: List[str],
                         categorical_features: List[str],
                         target_variable: str,
                         features_col: str,
                         with_std: bool = True,
                         with_mean: bool = False,
                         k_fold: int = 5) -> CrossValidator:
    
    data_encoder = encoding_pipeline(dataframe,
                                   numerical_features,
                                   categorical_features,
                                   target_variable,
                                   with_std,
                                   with_mean)
    
    classifier = RandomForestClassifier(featuresCol=features_col, 
                                      labelCol=LABEL_COL)
    
    predictions_idx_to_str = IndexToString(inputCol=PREDICTION_COL,
                                         outputCol=PREDICTED_TARGET_VARIABLE_COL,
                                         labels=data_encoder.fit(dataframe).stages[0].labels)
    
    stages = [data_encoder, classifier, predictions_idx_to_str]
    pipeline = Pipeline(stages=stages)
    
    param_grid = ParamGridBuilder() \
        .addGrid(classifier.maxDepth, [3, 5, 8]) \
        .addGrid(classifier.numTrees, [10, 50, 100]) \
        .build()
    
    evaluator = MulticlassClassificationEvaluator(labelCol=LABEL_COL,
                                                predictionCol=PREDICTION_COL,
                                                metricName='accuracy')
    
    cross_val = CrossValidator(estimator=pipeline,
                             estimatorParamMaps=param_grid,
                             evaluator=evaluator,
                             numFolds=k_fold,
                             collectSubModels=True)
    
    return cross_val.fit(dataframe)

# Load datasets
print("Loading datasets...")
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

# Process and combine measurements
print("Processing measurements...")
weather_measurements_df = None

for _, row in city_attributes_df.toPandas().iterrows():
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

# Clean data
print("Cleaning data...")
df = weather_measurements_df.na.drop()

# Add temporal features
df = df.withColumn("hour", hour(col(DATETIME_COL))) \
       .withColumn("month", month(col(DATETIME_COL)))

# Cast numeric columns
numeric_cols = [HUMIDITY_COL, PRESSURE_COL, TEMPERATURE_COL, 
                WIND_DIRECTION_COL, WIND_SPEED_COL, 
                LATITUDE_COL, LONGITUDE_COL,
                "hour", "month"]

for col_name in numeric_cols:
    df = df.withColumn(col_name, col(col_name).cast('double'))

# Define features
numerical_features = [
    HUMIDITY_COL, PRESSURE_COL, TEMPERATURE_COL,
    WIND_DIRECTION_COL, WIND_SPEED_COL,
    LATITUDE_COL, LONGITUDE_COL,
    "hour", "month"
]

categorical_features = [CITY_COL, COUNTRY_COL]

# Split data
print("Splitting data...")
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Train model
print("Training model...")
cv_model = random_forest_pipeline(
    dataframe=train_data,
    numerical_features=numerical_features,
    categorical_features=categorical_features,
    target_variable=WEATHER_CONDITION_COL,
    features_col=FEATURES_COL,
    with_std=True,
    with_mean=True,
    k_fold=5
)

# Make predictions
print("\nMaking predictions...")
predictions = cv_model.transform(test_data)

# Evaluate model
print("\n=== Model Performance Metrics ===")

# Calculate all metrics
evaluator_accuracy = MulticlassClassificationEvaluator(
    labelCol=LABEL_COL,
    predictionCol=PREDICTION_COL,
    metricName="accuracy"
)
accuracy = evaluator_accuracy.evaluate(predictions)
print(f"\nAccuracy: {accuracy:.4f}")

metrics = ['weightedPrecision', 'weightedRecall', 'f1']
for metric in metrics:
    evaluator = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL,
        predictionCol=PREDICTION_COL,
        metricName=metric
    )
    score = evaluator.evaluate(predictions)
    print(f"{metric}: {score:.4f}")

# Show class distribution
print("\nPer-Class Performance:")
predictions.groupBy(PREDICTED_TARGET_VARIABLE_COL).count().show()

# Get best model parameters
best_rf_model = cv_model.bestModel.stages[1]
print("\nBest Model Parameters:")
print(f"Max Depth: {best_rf_model.getMaxDepth()}")
print(f"Num Trees: {best_rf_model.getNumTrees()}")

# Feature importance
feature_names = numerical_features + categorical_features
importances = [(feature, float(importance)) 
               for feature, importance in zip(feature_names, best_rf_model.featureImportances)]
importances.sort(key=lambda x: x[1], reverse=True)

print("\nTop 5 Most Important Features:")
for feature, importance in importances[:5]:
    print(f"{feature}: {importance:.4f}")

# Final summary
print("\n=== Final Model Performance Summary ===")
print(f"Final Model Accuracy: {accuracy:.4f}")
print("Training Complete!")

# Save model
model_path = "hdfs://namenode:8020/model/weather_prediction_model_rf"
cv_model.bestModel.write().overwrite().save(model_path)
print(f"\nModel saved to: {model_path}")

# Stop Spark session
spark.stop()

