import streamlit as st
import pandas as pd

months = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

def get_next_month(months: list, current_month: str) -> str:
    """Returns the full name of the next month given the current month."""
    try:
        idx = months.index(current_month)
        next_idx = (idx + 1) % 12
        return months[next_idx]
    except ValueError:
        raise ValueError(f"Invalid month name: {current_month}. Please provide a full month name like 'March'.")


st.title("📦 Kola Inventory Recommendation Tool")

# # Load products from CSV
# uploaded_file = st.file_uploader("Upload a CSV file with 'product_name' column", type="csv")

trade_features = pd.read_csv('trade_import_2.csv')
prod_mapping = pd.read_csv('product_mapped_cat_gd.csv')

product = st.selectbox("Select a product", prod_mapping['Product Name'].unique())
prev_month = st.selectbox("Select the previous month", months)
units = st.number_input("Enter previous month's sales (units)", min_value=1)

if st.button("Get Recommendation"):
    pred_month = get_next_month(months, prev_month)
    category = list(prod_mapping[prod_mapping['Product Name']==product].drop_duplicates()['Imp_category'])[0]
    trade = trade_features[trade_features['Description_HS4']==category][['date','Month','Description_HS4','value_MoM_pct_change']]
    pct_change = list(trade[trade['Month']==pred_month]['value_MoM_pct_change'])[0]
    print(pct_change)
    recommended_units = round(units * (1 + pct_change))

    st.success(f"📊 Recommended units for **{product}** in **{pred_month}**: **{recommended_units} units**")
