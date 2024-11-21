import os
import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import plotly.express as px
import atexit

# Environment variables
os.environ['JAVA_HOME'] = '/opt/bitnami/java'
os.environ['SPARK_HOME'] = '/opt/bitnami/spark'
os.environ['HADOOP_HOME'] = '/opt/bitnami/hadoop'

# Initialize Spark Session
if 'spark' not in st.session_state:
    st.session_state.spark = SparkSession.builder \
        .appName("Weather Prediction Dashboard") \
        .getOrCreate()

spark = st.session_state.spark
spark.sparkContext.setLogLevel("WARN")

# Load model and data
model_path = "hdfs://namenode:8020/input/Makefile/model/weather_gbt_model"
data_path = "hdfs://namenode:8020/input/Makefile/data/weather_preprocessed_data_1"

try:
    pipeline_model = PipelineModel.load(model_path)
    st.session_state['model'] = pipeline_model
    df_spark = spark.read.parquet(data_path)
    df = df_spark.toPandas()
except Exception as e:
    st.error(f"Failed to load model or data: {e}")
    st.stop()

# Custom CSS
st.markdown("""
    <style>
    body {
        font-family: 'Arial', sans-serif;
        background-color: #f9f9f9;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 8px 16px;
        text-align: center;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 5px;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f8f9fa;
        margin: 20px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Navigation
page = st.sidebar.selectbox("Choose a page:", ["Weather Prediction", "Weather Analysis"])

if page == "Weather Prediction":
    st.title("Weather Condition Prediction")

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
            input_data = pd.DataFrame([[
                humidity, pressure, temperature, wind_direction,
                wind_speed, latitude, longitude, hour, month
            ]], columns=[
                'humidity', 'pressure', 'temperature', 'wind_direction',
                'wind_speed', 'latitude', 'longitude', 'hour', 'month'
            ])
            
            input_spark_df = spark.createDataFrame(input_data)
            prediction = pipeline_model.transform(input_spark_df)
            predicted_weather = prediction.select("prediction").collect()[0][0]

            weather_conditions = ['sunny', 'rainy', 'cloudy', 'foggy', 'snowy', 'thunderstorm']
            predicted_weather_condition = weather_conditions[int(predicted_weather)]

            weather_emojis = {
                'sunny': '☀️', 'rainy': '🌧️', 'cloudy': '☁️',
                'foggy': '🌫️', 'snowy': '🌨️', 'thunderstorm': '⛈️'
            }

            emoji = weather_emojis.get(predicted_weather_condition.lower(), '🌤️')
            
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.write(f"### Predicted Weather: {emoji} {predicted_weather_condition.title()}")
            st.markdown('</div>', unsafe_allow_html=True)

            # Display input summary
            st.write("### Input Summary")
            summary_df = pd.DataFrame({
                'Parameter': input_data.columns,
                'Value': input_data.iloc[0].values
            })
            st.table(summary_df)

        except Exception as e:
            st.error(f"Prediction error: {e}")

elif page == "Weather Analysis":
    st.title("Weather Data Analysis")

    # Sidebar filters
    st.sidebar.subheader("Filters")
    temp_range = st.sidebar.slider("Temperature Range (K)", 
                                 float(df['temperature'].min()), 
                                 float(df['temperature'].max()), 
                                 (float(df['temperature'].min()), float(df['temperature'].max())))
    
    selected_months = st.sidebar.multiselect("Select Months", 
                                           sorted(df['month'].unique()),
                                           sorted(df['month'].unique()))

    # Filter data
    filtered_df = df[
        (df['temperature'].between(temp_range[0], temp_range[1])) &
        (df['month'].isin(selected_months))
    ]

    # Display statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Temperature (K)", f"{filtered_df['temperature'].mean():.2f}")
    with col2:
        st.metric("Average Humidity (%)", f"{filtered_df['humidity'].mean():.2f}")
    with col3:
        st.metric("Average Pressure (hPa)", f"{filtered_df['pressure'].mean():.2f}")

    # Plots
    col1, col2 = st.columns(2)
    
    with col1:
        temp_hist = px.histogram(filtered_df, x='temperature', 
                               title="Temperature Distribution",
                               labels={'temperature': 'Temperature (K)'})
        st.plotly_chart(temp_hist)

        humid_scatter = px.scatter(filtered_df, x='temperature', y='humidity',
                                 title="Temperature vs Humidity",
                                 labels={'temperature': 'Temperature (K)', 
                                        'humidity': 'Humidity (%)'})
        st.plotly_chart(humid_scatter)

    with col2:
        wind_scatter = px.scatter(filtered_df, x='wind_speed', y='wind_direction',
                                title="Wind Analysis",
                                labels={'wind_speed': 'Wind Speed (m/s)', 
                                       'wind_direction': 'Wind Direction (degrees)'})
        st.plotly_chart(wind_scatter)

        pressure_box = px.box(filtered_df, x='month', y='pressure',
                            title="Pressure Distribution by Month",
                            labels={'month': 'Month', 'pressure': 'Pressure (hPa)'})
        st.plotly_chart(pressure_box)

# Cleanup
@atexit.register
def cleanup():
    if 'spark' in st.session_state:
        st.session_state.spark.stop()


