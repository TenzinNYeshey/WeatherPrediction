import os
import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler, StringIndexer
import plotly.express as px
import atexit

# Set environment variables
os.environ['JAVA_HOME'] = '/opt/bitnami/java'
os.environ['SPARK_HOME'] = '/opt/bitnami/spark'
os.environ['HADOOP_HOME'] = '/opt/bitnami/hadoop'

# Initialize Spark Session
if 'spark' not in st.session_state:
    st.session_state.spark = SparkSession.builder \
        .appName("Weather Prediction Dashboard") \
        .getOrCreate()

# Set Spark log level
spark = st.session_state.spark
spark.sparkContext.setLogLevel("WARN")

# Load the trained Random Forest model
model_path = "hdfs://namenode:8020/model/weather_prediction_model"
try:
    rf_model = RandomForestClassificationModel.load(model_path)
    st.session_state['model'] = rf_model
except Exception as e:
    st.error(f"Failed to load Weather Prediction model: {e}")
    st.stop()

# Load preprocessed data
data_path = "hdfs://namenode:8020/model/weather_preprocessed_data"
try:
    df_spark = spark.read.parquet(data_path)
    df = df_spark.toPandas()
except Exception as e:
    st.error(f"Failed to load preprocessed data: {e}")
    st.stop()

# Custom CSS
st.markdown("""
    <style>
        .reportview-container {
            background-color: #f0f8ff;
        }
        .sidebar .sidebar-content {
            background-color: #1e3d59;
            color: white;
        }
        .stButton>button {
            background-color: #17b794;
            color: white;
            font-size: 16px;
            border-radius: 10px;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background-color: #148f77;
        }
        h1, h2, h3, h4 {
            font-family: 'Arial', sans-serif;
            color: #2c3e50;
        }
    </style>
""", unsafe_allow_html=True)

# Navigation Menu
st.sidebar.title("Weather Prediction")
page = st.sidebar.selectbox("Choose a page:", ["Weather Prediction", "Weather Analysis"])

if page == "Weather Prediction":
    st.title("Weather Condition Prediction")

    # Input fields for prediction
    col1, col2 = st.columns(2)
    with col1:
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=1.0)
        pressure = st.number_input("Pressure (hPa)", min_value=800.0, max_value=1100.0, step=1.0)
        temperature = st.number_input("Temperature (K)", min_value=240.0, max_value=320.0, step=0.1)
        wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=50.0, step=0.1)

    with col2:
        wind_direction = st.number_input("Wind Direction (degrees)", min_value=0.0, max_value=360.0, step=1.0)
        latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, step=0.1)
        longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, step=0.1)
        hour = st.number_input("Hour of Day", min_value=0, max_value=23, step=1)
        month = st.number_input("Month", min_value=1, max_value=12, step=1)

    if st.button("🌤️ Predict Weather"):
        try:
            # Create input DataFrame
            input_data = pd.DataFrame([[
                humidity, pressure, temperature, wind_direction, 
                wind_speed, latitude, longitude, hour, month
            ]], columns=[
                'humidity', 'pressure', 'temperature', 'wind_direction',
                'wind_speed', 'latitude', 'longitude', 'hour', 'month'
            ])
            
            # Convert to Spark DataFrame
            input_spark_df = spark.createDataFrame(input_data)

            # Create feature vector
            assembler = VectorAssembler(
                inputCols=['humidity', 'pressure', 'temperature', 'wind_direction',
                          'wind_speed', 'latitude', 'longitude', 'hour', 'month'],
                outputCol="features")
            
            input_spark_df = assembler.transform(input_spark_df)
            
            # Make prediction
            prediction = rf_model.transform(input_spark_df)
            predicted_weather = prediction.select("predicted_weather").collect()[0][0]
            
            # Display prediction with emoji
            weather_emojis = {
                'sunny': '☀️',
                'rainy': '🌧️',
                'cloudy': '☁️',
                'foggy': '🌫️',
                'snowy': '🌨️',
                'thunderstorm': '⛈️'
            }
            
            emoji = weather_emojis.get(predicted_weather.lower(), '')
            st.write(f"### Predicted Weather: {emoji} {predicted_weather}")

            # Display confidence scores
            probabilities = prediction.select("probability").collect()[0][0]
            st.write("### Confidence Scores:")
            for weather, prob in zip(rf_model.stages[0].labels, probabilities):
                st.write(f"{weather}: {prob:.2%}")

        except Exception as e:
            st.error(f"Prediction error: {e}")

elif page == "Weather Analysis":
    st.title("Weather Data Analysis")
    
    # Filters
    st.sidebar.header("Filters")
    selected_months = st.sidebar.multiselect("Select Months", 
                                           options=sorted(df['month'].unique()), 
                                           default=sorted(df['month'].unique()))

    # Filter data
    filtered_df = df[df['month'].isin(selected_months)]

    col1, col2 = st.columns(2)

    with col1:
        # Temperature distribution
        st.write("Temperature Distribution by Weather")
        temp_fig = px.box(filtered_df, 
                         x='weather_condition', 
                         y='temperature',
                         title="Temperature Distribution by Weather Condition")
        st.plotly_chart(temp_fig)

        # Wind speed vs direction
        st.write("Wind Speed vs Direction")
        wind_fig = px.scatter_polar(filtered_df, 
                                  r="wind_speed",
                                  theta="wind_direction",
                                  color="weather_condition",
                                  title="Wind Pattern Analysis")
        st.plotly_chart(wind_fig)

    with col2:
        # Weather condition distribution
        st.write("Weather Condition Distribution")
        weather_dist = filtered_df['weather_condition'].value_counts()
        pie_fig = px.pie(values=weather_dist.values,
                        names=weather_dist.index,
                        title="Distribution of Weather Conditions")
        st.plotly_chart(pie_fig)

        # Humidity vs Temperature
        st.write("Humidity vs Temperature")
        humid_temp_fig = px.scatter(filtered_df,
                                  x="temperature",
                                  y="humidity",
                                  color="weather_condition",
                                  title="Humidity vs Temperature")
        st.plotly_chart(humid_temp_fig)

    # Additional Statistics
    st.write("### Weather Statistics")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.metric("Average Temperature (K)", 
                 f"{filtered_df['temperature'].mean():.2f}")
    with col4:
        st.metric("Average Humidity (%)", 
                 f"{filtered_df['humidity'].mean():.2f}")
    with col5:
        st.metric("Average Pressure (hPa)", 
                 f"{filtered_df['pressure'].mean():.2f}")

# Cleanup
atexit.register(lambda: spark.stop() if 'spark' in st.session_state else None)

