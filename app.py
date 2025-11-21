import streamlit as st
import pandas as pd
import pickle
import os
from math import radians, sin, cos, sqrt, atan2
from sklearn.preprocessing import LabelEncoder

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def preprocess_data(df):
    df = df.drop(columns=['Unnamed: 0', 'cc_num', 'first', 'last', 'street', 'city', 'state', 'zip', 'trans_num', 'unix_time', 'merchant', 'job'], errors='ignore')
    
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
    df['month'] = df['trans_date_trans_time'].dt.month
    df = df.drop(columns=['trans_date_trans_time'])
    
    df['dob'] = pd.to_datetime(df['dob'])
    df['age'] = 2025 - df['dob'].dt.year
    df = df.drop(columns=['dob'])
    
    df['distance'] = df.apply(lambda row: haversine(row['lat'], row['long'], row['merch_lat'], row['merch_long']), axis=1)
    df = df.drop(columns=['lat', 'long', 'merch_lat', 'merch_long'])
    
    le = LabelEncoder()
    df['category'] = le.fit_transform(df['category'])
    df['gender'] = le.fit_transform(df['gender'])
    
    return df

@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

st.title("📊 Batch Fraud Detection")

if not os.path.exists('model.pkl') or not os.path.exists('scaler.pkl'):
    st.error("⚠️ Model not trained yet. Please run: `python train_model.py`")
    st.stop()

model, scaler = load_model()

uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    st.write(f"Loaded {len(df_raw)} transactions")
    
    if st.button("🔍 Predict All"):
        with st.spinner("Processing..."):
            df_processed = preprocess_data(df_raw.copy())
            
            if 'is_fraud' in df_processed.columns:
                df_processed = df_processed.drop('is_fraud', axis=1)
            
            df_scaled = scaler.transform(df_processed)
            predictions = model.predict(df_scaled)
            probabilities = model.predict_proba(df_scaled)[:, 1]
            
            df_raw['Prediction'] = predictions
            df_raw['Fraud_Probability'] = probabilities
            
            fraud_count = predictions.sum()
            st.metric("Fraudulent Transactions", f"{fraud_count} ({fraud_count/len(df_raw)*100:.1f}%)")
            
            st.dataframe(df_raw[df_raw['Prediction'] == 1])