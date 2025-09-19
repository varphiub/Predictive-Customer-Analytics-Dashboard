# 02_clv_modeling.py

import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter

print("Starting CLV modeling...")

# Load the RFM data
try:
    rfm_df = pd.read_csv('customer_rfm_data.csv', index_col='CustomerID')
    print("RFM data loaded successfully.")
except FileNotFoundError:
    print("Error: customer_rfm_data.csv not found. Please run 01_process_data.py first.") 
    exit()

# --- WEEK 3: CLV MODELING ---

# BG/NBD Model (to predict purchase frequency)
bgf = BetaGeoFitter(penalizer_coef=0.001) # Small penalizer to prevent overfitting
bgf.fit(rfm_df['Frequency'], rfm_df['Recency'], rfm_df['T'])

print("\n--- BG/NBD Model Summary ---")
print(bgf.summary)

# Gamma-Gamma Model (to predict monetary value)
# Prerequisite: Check that monetary value and frequency are not highly correlated
correlation = rfm_df[['Monetary', 'Frequency']].corr().iloc[0, 1]
print(f"\nCorrelation between Monetary and Frequency: {correlation:.4f}")

# Filter out customers with zero monetary value as they are not purchases
ggf_df = rfm_df[rfm_df['Monetary'] > 0]

ggf = GammaGammaFitter(penalizer_coef=0.001)
ggf.fit(ggf_df['Frequency'], ggf_df['Monetary'])

print("\n--- Gamma-Gamma Model Summary ---")
print(ggf.summary)

# Calculate CLV for the next 90 days (3 months)
clv_predictions = ggf.customer_lifetime_value(
    bgf,
    rfm_df['Frequency'],
    rfm_df['Recency'],
    rfm_df['T'],
    rfm_df['Monetary'],
    time=3,  # 3 months
    discount_rate=0.01 # Monthly discount rate
)

# Merge CLV predictions back into the main dataframe
rfm_df['predicted_clv_90_days'] = clv_predictions

# Save the data with CLV predictions
output_path = 'customer_clv_data.csv'
rfm_df.to_csv(output_path)

print(f"\nCLV modeling complete. Data with CLV predictions saved to {output_path}")
print("\n--- Sample of data with CLV predictions ---")
print(rfm_df.sort_values(by='predicted_clv_90_days', ascending=False).head())