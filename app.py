# app.py

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Customer Analytics Dashboard",
    page_icon="📈",
    layout="wide"
)

# --- LOAD DATA AND MODELS ---
@st.cache_data
def load_data():
    """Loads the final customer data with CLV and churn info."""
    try:
        data = pd.read_csv('customer_clv_data.csv')
        return data
    except FileNotFoundError:
        st.error("Data file 'customer_clv_data.csv' not found. Please run the modeling scripts first.")
        return None

@st.cache_resource
def load_models():
    """Loads the trained churn model and scaler."""
    try:
        model = joblib.load('churn_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Model or scaler file not found. Please run the churn modeling script.")
        return None, None

df = load_data()
churn_model, scaler = load_models()

if df is None or churn_model is None:
    st.stop()

# Add churn probability to the dataframe
features = ['Recency', 'Frequency', 'Monetary', 'T']
df_scaled = scaler.transform(df[features])
df['churn_probability'] = churn_model.predict_proba(df_scaled)[:, 1]


# --- DASHBOARD UI ---
st.title("📈 Predictive Customer Analytics Dashboard")
st.markdown("An interactive dashboard to analyze customer value and churn risk.")

# --- KPIs ---
st.header("Overall Customer Insights")
total_customers = df['CustomerID'].nunique()
avg_clv = df['predicted_clv_90_days'].mean()
churn_rate = df[df['churn_probability'] > 0.5]['CustomerID'].nunique() / total_customers

col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Average 90-Day CLV", f"${avg_clv:,.2f}")
col3.metric("Predicted Churn Rate", f"{churn_rate:.2%}")

st.markdown("---")

# --- CUSTOMER SEGMENTATION AND ANALYSIS ---
st.header("Customer Segmentation")
col1, col2 = st.columns(2)

with col1:
    st.subheader("RFM Distribution")
    fig_rfm = px.scatter_3d(
        df, x='Recency', y='Frequency', z='Monetary',
        color='predicted_clv_90_days',
        title="RFM 3D Scatter Plot",
        hover_data=['CustomerID'],
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_rfm, use_container_width=True)

with col2:
    st.subheader("CLV vs. Churn Probability")
    fig_clv_churn = px.scatter(
        df, x='predicted_clv_90_days', y='churn_probability',
        color='churn_probability',
        title="Customer Value vs. Churn Risk",
        hover_data=['CustomerID', 'Recency', 'Frequency'],
        color_continuous_scale='Reds'
    )
    st.plotly_chart(fig_clv_churn, use_container_width=True)

st.markdown("---")

# --- INDIVIDUAL CUSTOMER LOOKUP ---
st.header("Individual Customer Lookup")
customer_id = st.selectbox(
    "Select a Customer ID to analyze:",
    df['CustomerID'].unique()
)

if customer_id:
    customer_data = df[df['CustomerID'] == customer_id].iloc[0]
    st.subheader(f"Analysis for Customer: {int(customer_data['CustomerID'])}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recency", f"{customer_data['Recency']} days")
    c2.metric("Frequency", f"{customer_data['Frequency']} purchases")
    c3.metric("Monetary", f"${customer_data['Monetary']:,.2f}")
    c4.metric("Tenure (T)", f"{int(customer_data['T'])} days")

    clv_val = customer_data['predicted_clv_90_days']
    churn_prob = customer_data['churn_probability']

    st.metric("Predicted 90-Day CLV", f"${clv_val:,.2f}")
    
    st.progress(churn_prob, text=f"Churn Probability: {churn_prob:.2%}")
    if churn_prob > 0.75:
        st.error("🚨 High Churn Risk: Immediate action recommended.")
    elif churn_prob > 0.5:
        st.warning("⚠️ Medium Churn Risk: Consider retention campaign.")
    else:
        st.success("✅ Low Churn Risk: Customer is likely loyal.")

st.markdown("---")

# --- HIGH-RISK CUSTOMER LIST ---
st.header("High-Risk Customer Segments")
risk_threshold = st.slider("Select churn probability threshold:", 0.0, 1.0, 0.75)
high_risk_df = df[df['churn_probability'] >= risk_threshold].sort_values(by='churn_probability', ascending=False)

st.subheader(f"Customers with > {risk_threshold:.0%} Churn Probability")
st.dataframe(
    high_risk_df[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'predicted_clv_90_days', 'churn_probability']],
    use_container_width=True
)