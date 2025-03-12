import streamlit as st
import pandas as pd
import pickle
import numpy as np
from datetime import datetime

# Load trained XGBoost model (ensure the model is saved)
@st.cache_resource
def load_model():
    with open("linear_reg_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Load historical data (for lag feature calculation)
@st.cache_data
def load_data_pipeline():
    sales_data = pd.read_csv("Kola_Sales_20250225.csv", parse_dates=["created_at"])
    sales_data = sales_data[(sales_data['created_at'] >= '2024-01-01') & (sales_data['created_at'] < '2025-03-01')]
    sales_data = sales_data[['created_at','name','product_name','quantity']]
    sales_data["created_at"] = pd.to_datetime(sales_data["created_at"])
    # Extract Month-Year in the format "MM-YYYY"
    sales_data["month_year"] = sales_data["created_at"].dt.strftime("%m-%Y")
    # Get latest recorded month per customer per product
    latest_records = sales_data.groupby(["name", "product_name"]).agg(
        max_date=("created_at", "max")
    ).reset_index()
    latest_records["Latest_Month_Year"] = latest_records["max_date"].dt.strftime("%m-%Y")
    # latest_records
    # Get total quantity for the latest month per product
    total_quantity_per_group = sales_data.groupby(["name","product_name", "month_year"])["quantity"].sum().reset_index()
    total_quantity_per_group.rename(columns={"quantity": "Total_Quantity_in_Max_Month"}, inplace=True)
    # total_quantity_per_group

    # Merge to get total quantity in the latest month per name and product
    latest_records = latest_records.merge(
        total_quantity_per_group, 
        how="left", 
        left_on=["name", "product_name", "Latest_Month_Year"], 
        right_on=["name", "product_name", "month_year"]
    ).drop(columns=["month_year"]).rename(columns={"Total_Quantity": "Total_Quantity_in_Max_Month"})
    # latest_records

    def get_prev_month_quantity(name, product, max_month_year, months_ago):
        max_month_dt = pd.to_datetime(max_month_year, format="%m-%Y")
        target_month = max_month_dt - pd.DateOffset(months=months_ago)
        target_month_str = target_month.strftime("%m-%Y")
        
        prev_month_data = sales_data[
            (sales_data["name"] == name) & 
            (sales_data["product_name"] == product) & 
            (sales_data["month_year"] == target_month_str)
        ]
        # print(prev_month_data)
        return prev_month_data["quantity"].sum()
    
    # Apply function to get total quantity for the previous 6 months
    latest_records["Quantity_Month_1_Ago"] = latest_records.apply(
        lambda row: get_prev_month_quantity(row["name"], row["product_name"], row["Latest_Month_Year"], 1), axis=1
    )
    latest_records["Quantity_Month_2_Ago"] = latest_records.apply(
        lambda row: get_prev_month_quantity(row["name"], row["product_name"], row["Latest_Month_Year"], 2), axis=1
    )
    latest_records["Quantity_Month_3_Ago"] = latest_records.apply(
        lambda row: get_prev_month_quantity(row["name"], row["product_name"], row["Latest_Month_Year"], 3), axis=1
    )
    latest_records["Quantity_Month_4_Ago"] = latest_records.apply(
        lambda row: get_prev_month_quantity(row["name"], row["product_name"], row["Latest_Month_Year"], 4), axis=1
    )
    latest_records["Quantity_Month_5_Ago"] = latest_records.apply(
        lambda row: get_prev_month_quantity(row["name"], row["product_name"], row["Latest_Month_Year"], 5), axis=1
    )
    latest_records["Quantity_Month_6_Ago"] = latest_records.apply(
        lambda row: get_prev_month_quantity(row["name"], row["product_name"], row["Latest_Month_Year"], 6), axis=1
    )
    return latest_records

df = load_data_pipeline()

# Streamlit UI
st.title("🤖📦 Kola Inventory Manager - KIM")

# Get unique names and products
selected_name = st.selectbox("Select Customer Name", df["name"].unique())
selected_product = st.selectbox("Select Product", df[df["name"]==selected_name]["product_name"].unique())

# filter on selected name 
filtered_df = df[(df["name"] == selected_name) & (df["product_name"] == selected_product)]
filtered_df


# Calendar input for selecting prediction month & year
selected_date = st.date_input("Select Target Month & Year", datetime.today())
month = selected_date.month
year = selected_date.year

quantity_1 = filtered_df['Total_Quantity_in_Max_Month']
quantity_2 = filtered_df['Quantity_Month_1_Ago']
quantity_3 = filtered_df['Quantity_Month_2_Ago']
quantity_4 = filtered_df['Quantity_Month_3_Ago']
quantity_5 = filtered_df['Quantity_Month_4_Ago']
quantity_6 = filtered_df['Quantity_Month_5_Ago']

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

print(input_data.head())
    
# Predict demand
if st.button("Predict Demand"):
    prediction = model.predict(input_data)
    print(prediction)
    st.success(f"📈 Predicted Demand for {selected_product} in {month}/{year}: **{round(prediction[0])-21} units**")

