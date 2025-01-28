# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Page Configuration
st.set_page_config(
    page_title="Air Quality Analysis",
    layout="wide"
)

# Data Loading and Preparation
@st.cache_data
def load_data():
    data = pd.read_csv("./merged_air_quality.csv")
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)
    return data

df = load_data()

# Title
st.title("Air Quality Analysis")
st.write("Welcome to the Air Quality Analysis app!")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", [
    "Data Overview",
    "Exploratory Data Analysis",
    "Modeling & Prediction"
])

# Data Overview Function
def data_overview():
    st.title("Air Quality Data Overview")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset Summary")
        st.write(f"Total Records: {df.shape[0]}")
        st.write(f"Total Features: {df.shape[1]}")
        st.write("Time Range:", df.index.min().date(), "to", df.index.max().date())
    
    with col2:
        st.subheader("Missing Values")
        missing = df.isnull().sum().to_frame("Missing Values")
        st.dataframe(missing.style.highlight_null(color='#FF9999'))
    
    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("Column Descriptions")
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes,
        'Unique Values': [df[col].nunique() for col in df.columns]
    })
    st.table(col_info)

# Main App Controller
if page == "Data Overview":
    data_overview()
else:
    st.write("Not implemented yet!")

