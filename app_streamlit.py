import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load model
model = joblib.load("disease_model.pkl")

# Load dataset
df = pd.read_csv("Training.csv")
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Identify target column
if "prognosis" in df.columns:
    target_col = "prognosis"
else:
    target_col = "disease"

# Prepare features and labels
X = df.drop(columns=[target_col])
y = df[target_col]

# 🔥 Recreate label encoder (IMPORTANT FIX)
label_encoder = LabelEncoder()
label_encoder.fit(y)

symptoms = X.columns

# UI
st.title("Disease Prediction System")

selected_symptoms = st.multiselect("Select Symptoms", symptoms)

# Create input
input_data = {col: 0 for col in symptoms}

for s in selected_symptoms:
    input_data[s] = 1

input_df = pd.DataFrame([input_data])

# Prediction
if st.button("Predict"):
    pred_label = model.predict(input_df)[0]

    # 🔥 Convert number → disease name
    pred_disease = label_encoder.inverse_transform([pred_label])[0]

    st.success(f"Predicted Disease: {pred_disease}")