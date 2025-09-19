# 03_churn_modeling.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

print("Starting churn prediction modeling...")

# Load data with features (we can use the original RFM file)
try:
    rfm_df = pd.read_csv('customer_rfm_data.csv', index_col='CustomerID') 
    print("RFM data loaded successfully.")
except FileNotFoundError:
    print("Error: customer_rfm_data.csv not found. Please run 01_process_data.py first.")
    exit()

# --- WEEK 4: CHURN PREDICTION ---

# Task 1: Prepare Data for ML
features = ['Recency', 'Frequency', 'Monetary', 'T']
target = 'is_churned'

X = rfm_df[features]
y = rfm_df[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

print(f"\nOriginal training data shape: {X_train.shape}")
print(f"Resampled training data shape: {X_train_resampled.shape}")
print("Class distribution after SMOTE:")
print(pd.Series(y_train_resampled).value_counts())

# Task 2: Train XGBoost Model
print("\nTraining XGBoost model...")
xgb_model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)
xgb_model.fit(X_train_resampled, y_train_resampled)

# Task 3: Evaluate Model
print("\nEvaluating model performance...")
y_pred = xgb_model.predict(X_test_scaled)
y_pred_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

print(f"\n--- ROC AUC Score ---")
print(f"{roc_auc_score(y_test, y_pred_proba):.4f}")

# Save the trained model and the scaler
joblib.dump(xgb_model, 'churn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nChurn model and scaler have been saved as 'churn_model.pkl' and 'scaler.pkl'")