import pandas as pd
import numpy as np
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, lit, hour, month
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler, OneHotEncoder
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from typing import List
import os

# Set environment variables
os.environ['JAVA_HOME'] = '/opt/bitnami/java'
os.environ['SPARK_HOME'] = '/opt/bitnami/spark'

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("WeatherPrediction") \
    .config("spark.executor.memory", "16g") \
    .config("spark.driver.memory", "8g") \
    .config("spark.sql.shuffle.partitions", "200") \
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
LABEL_COL = 'label'
FEATURES_COL = 'features'

# Encoding pipeline
def create_pipeline(numerical_features: List[str], categorical_features: List[str], target_col: str) -> Pipeline:
    stages = []

    # StringIndexer for target column
    label_indexer = StringIndexer(inputCol=target_col, outputCol=LABEL_COL, handleInvalid="skip")
    stages.append(label_indexer)

    # Encode categorical features
    for cat_col in categorical_features:
        string_indexer = StringIndexer(inputCol=cat_col, outputCol=f"{cat_col}_index", handleInvalid="skip")
        encoder = OneHotEncoder(inputCol=f"{cat_col}_index", outputCol=f"{cat_col}_encoded")
        stages += [string_indexer, encoder]

    # Assemble features
    feature_cols = numerical_features + [f"{cat_col}_encoded" for cat_col in categorical_features]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="assembled_features")
    stages.append(assembler)

    # Scale features
    scaler = StandardScaler(inputCol="assembled_features", outputCol=FEATURES_COL, withStd=True, withMean=False)
    stages.append(scaler)

    return Pipeline(stages=stages)

# Load data
hdfs_path = "hdfs://namenode:8020/dataset/Makefile"
weather_df = spark.read.csv(hdfs_path, header=True, inferSchema=True)

# Filter necessary columns
columns_to_keep = [
    DATETIME_COL, HUMIDITY_COL, PRESSURE_COL, TEMPERATURE_COL, WIND_DIRECTION_COL, WIND_SPEED_COL,
    LATITUDE_COL, LONGITUDE_COL, CITY_COL, COUNTRY_COL, WEATHER_CONDITION_COL
]
weather_df = weather_df.select([col for col in columns_to_keep if col in weather_df.columns])

# Add temporal features
weather_df = weather_df.withColumn("hour", hour(col(DATETIME_COL))) \
                       .withColumn("month", month(col(DATETIME_COL)))

# Cast numerical features to double
numerical_features = [
    HUMIDITY_COL, PRESSURE_COL, TEMPERATURE_COL, WIND_DIRECTION_COL, WIND_SPEED_COL,
    LATITUDE_COL, LONGITUDE_COL, "hour", "month"
]
for feature in numerical_features:
    weather_df = weather_df.withColumn(feature, col(feature).cast("double"))

# Split data
train_data, test_data = weather_df.randomSplit([0.7, 0.3], seed=42)

# Create pipeline
categorical_features = [CITY_COL, COUNTRY_COL]
pipeline = create_pipeline(numerical_features, categorical_features, WEATHER_CONDITION_COL)

# Train RandomForest model
rf_classifier = RandomForestClassifier(featuresCol=FEATURES_COL, labelCol=LABEL_COL, numTrees=50, maxDepth=5)
pipeline_model = Pipeline(stages=[pipeline, rf_classifier])

# Parameter tuning
param_grid = ParamGridBuilder() \
    .addGrid(rf_classifier.numTrees, [10, 50]) \
    .addGrid(rf_classifier.maxDepth, [3, 5]) \
    .build()

evaluator = MulticlassClassificationEvaluator(labelCol=LABEL_COL, predictionCol="prediction", metricName="accuracy")
cross_validator = CrossValidator(estimator=pipeline_model, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=3)

# Train and evaluate
cv_model = cross_validator.fit(train_data)
predictions = cv_model.transform(test_data)

# Evaluate model
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy:.4f}")

# Feature importance
rf_model = cv_model.bestModel.stages[-1]
feature_importances = rf_model.featureImportances
important_features = sorted(zip(feature_importances.indices, feature_importances.values), key=lambda x: x[1], reverse=True)

print("\nFeature Importance:")
for idx, importance in important_features[:5]:
    print(f"{numerical_features[idx]}: {importance:.4f}")

# Save best model
model_path = "hdfs://namenode:8020/model/weather_rf_model_try"
cv_model.bestModel.write().overwrite().save(model_path)

# Stop Spark session
spark.stop()
