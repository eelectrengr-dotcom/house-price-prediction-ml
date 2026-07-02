import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Page config
st.set_page_config(page_title="House Price App", page_icon="🏠", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .title {
        font-size:40px !important;
        font-weight:700;
        color:#2c3e50;
        text-align:center;
    }
    .subheader {
        font-size:20px !important;
        color:#34495e;
        text-align:center;
        margin-bottom:30px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        padding: 10px 24px;
    }
    </style>
""", unsafe_allow_html=True)

# Load dataset
data = pd.read_csv("data.csv")

X = data[['area', 'bedrooms', 'bathrooms']]
y = data['price']

# Train models
lr_model = LinearRegression()
lr_model.fit(X, y)

rf_model = RandomForestRegressor()
rf_model.fit(X, y)

# Title
st.markdown('<p class="title">🏠 House Price Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Enter details to estimate house price</p>', unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.header("Input Features")

area = st.sidebar.slider("Area (sq ft)", 500, 5000, 1500)
bedrooms = st.sidebar.slider("Bedrooms", 1, 6, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 5, 2)

model_choice = st.sidebar.selectbox("Select Model", ["Linear Regression", "Random Forest"])

# Predict button
if st.button("🔍 Predict Price"):
    input_data = [[area, bedrooms, bathrooms]]

    if model_choice == "Linear Regression":
        prediction = lr_model.predict(input_data)
    else:
        prediction = rf_model.predict(input_data)

    st.success(f"💰 Estimated Price: ${prediction[0]:,.2f}")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")