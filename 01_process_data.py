# 01_process_data.py

import pandas as pd
import datetime as dt

print("Starting data processing and feature engineering...")

# --- WEEK 1: FOUNDATION AND DATA DEEP DIVE ---

# Task 1 & 2: Load and Explore Data
# You might need to install openpyxl: pip install openpyxl
try:
    df = pd.read_excel(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx" 
    )
    print("Data loaded successfully.")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Task 3: Data Cleaning
print("Cleaning data...")
df.dropna(subset=['CustomerID'], inplace=True) # Drop rows with no customer ID
df = df[~df['InvoiceNo'].astype(str).str.contains('C')] # Remove cancellations
df = df[df['Quantity'] > 0] # Remove negative quantities
df = df[df['UnitPrice'] > 0] # Remove items with no price

# Convert CustomerID to integer for consistency
df['CustomerID'] = df['CustomerID'].astype(int)

# Create TotalPrice column
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
print("Data cleaning complete.")

# --- WEEK 2: FEATURE ENGINEERING (RFM) ---
print("Starting RFM feature engineering...")

# Set a snapshot date for RFM calculation (one day after the last transaction)
snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)

# Calculate RFM metrics
rfm_df = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda date: (snapshot_date - date.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
})

# Rename columns for clarity
rfm_df.rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'TotalPrice': 'Monetary'
}, inplace=True)

# Calculate Tenure (T)
customer_tenure = df.groupby('CustomerID')['InvoiceDate'].min().reset_index()
customer_tenure.rename(columns={'InvoiceDate': 'FirstPurchaseDate'}, inplace=True)
customer_tenure['T'] = (snapshot_date - customer_tenure['FirstPurchaseDate']).dt.days
rfm_df = rfm_df.join(customer_tenure.set_index('CustomerID')['T'])


# Define and create the Churn target variable
# If a customer's last purchase was more than 90 days ago, we'll consider them churned.
rfm_df['is_churned'] = rfm_df['Recency'].apply(lambda x: 1 if x > 90 else 0)

# Save the final analytical base table
output_path = 'customer_rfm_data.csv'
rfm_df.to_csv(output_path)

print(f"RFM feature engineering complete. Data saved to {output_path}")
print("\n--- Sample of the final data ---")
print(rfm_df.head())
print("\n--- Churn distribution ---")
print(rfm_df['is_churned'].value_counts(normalize=True))