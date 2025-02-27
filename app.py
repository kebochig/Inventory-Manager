import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load trained XGBoost model (ensure the model is saved)
@st.cache_resource
def load_model():
    with open("linear_reg_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Load historical data (for lag feature calculation)
@st.cache_data
def load_data():
    return pd.read_csv("Kola_Sales_20250225.csv", parse_dates=["created_at"])

df = load_data()

# Streamlit UI
st.title("🤖📦 Kola Inventory Manager - KIM")

# User input: Product selection
product_list = df["product_name"].unique()
product = st.selectbox("Select Product:", product_list)

# User input: Month and Year
month = st.slider("Select Month:", 1, 12, 6)
year = st.slider("Select Year:", 2024, 2026, 2025)

quantity_1 = st.number_input("Quantity (1 Month Ago)", min_value=0, value=100)
quantity_2 = st.number_input("Quantity (2 Months Ago)", min_value=0, value=120)
quantity_3 = st.number_input("Quantity (3 Months Ago)", min_value=0, value=90)
quantity_4 = st.number_input("Quantity (4 Months Ago)", min_value=0, value=80)
quantity_5 = st.number_input("Quantity (5 Months Ago)", min_value=0, value=70)
quantity_6 = st.number_input("Quantity (6 Months Ago)", min_value=0, value=60)

# Create input DataFrame
input_data = pd.DataFrame({
    "Quantity_Month_1_Ago": [quantity_1],
    "Quantity_Month_2_Ago": [quantity_2],
    "Quantity_Month_3_Ago": [quantity_3],
    "Quantity_Month_4_Ago": [quantity_4],
    "Quantity_Month_5_Ago": [quantity_5],
    "Quantity_Month_6_Ago": [quantity_6],
    "month": [month],
    "year": [year]
})
    
# Predict demand
if st.button("Predict Demand"):
    prediction = model.predict(input_data)
    print(prediction)
    st.success(f"📈 Predicted Demand for {product} in {month}/{year}: **{round(prediction[0])} units**")

