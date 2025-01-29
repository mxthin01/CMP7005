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

def eda():
    st.title("Exploratory Data Analysis")
    
    tab1, tab2, tab3 = st.tabs([
        "Univariate Analysis", 
        "Bivariate Analysis", 
        "Time Series Analysis"
    ])
    
    with tab1:
        st.subheader("Univariate Analysis")
        col = st.selectbox("Select Column", df.columns, key='uni_col')
        
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        sns.histplot(df[col], kde=True, ax=ax[0])
        ax[0].set_title(f"Distribution of {col}")
        sns.boxplot(x=df[col], ax=ax[1])
        ax[1].set_title(f"Boxplot of {col}")
        st.pyplot(fig)
    
    with tab2:
        st.subheader("Bivariate Analysis")
        col1 = st.selectbox("X-axis", df.columns, key='x_axis')
        col2 = st.selectbox("Y-axis", df.columns, key='y_axis')
        
        fig = px.scatter(
            df, x=col1, y=col2, 
            trendline="ols",
            title=f"{col1} vs {col2}"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Time Series Analysis")
        ts_col = st.selectbox("Select Metric", df.columns, key='ts_col')
        resample_freq = st.selectbox("Select Frequency", 
                                   ['D', 'W', 'M', 'Q', 'Y'], 
                                   index=2)
        
        ts_data = df[ts_col].resample(resample_freq).mean()
        fig = px.line(
            ts_data, 
            title=f"{ts_col} Over Time ({resample_freq} frequency)"
        )
        st.plotly_chart(fig, use_container_width=True)


def modeling():
    st.title("Air Quality Prediction Model")
    
    st.subheader("Model Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        target = st.selectbox("Select Target Variable", df.columns)
        features = st.multiselect("Select Features", df.columns.drop(target))
    
    with col2:
        test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2)
        n_estimators = st.slider("Number of Trees", 10, 200, 100)
        max_depth = st.slider("Max Depth", 2, 50, 20)
    
    if st.button("Train Model"):
        if not features:
            st.error("Please select at least one feature!")
            return
            
        X = df[features]
        y = df[target]
        
        # Train-test split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Model training
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Evaluation
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Display results
        st.subheader("Model Performance")
        col1, col2 = st.columns(2)
        col1.metric("Mean Squared Error", f"{mse:.2f}")
        col2.metric("R² Score", f"{r2:.2f}")
        
        # Feature importance
        st.subheader("Feature Importance")
        importance = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(
            importance, 
            x='Importance', 
            y='Feature', 
            orientation='h'
        )
        st.plotly_chart(fig, use_container_width=True)

# Update Main App Controller
if page == "Data Overview":
    data_overview()
elif page == "Exploratory Data Analysis":
    eda()
elif page == "Modeling & Prediction":
    modeling()