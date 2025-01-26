import streamlit as st
import pandas as pd
import joblib

# Load saved model, scaler, and encoders
rf_model = joblib.load('bet_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

st.title("Bet Safety Prediction Dashboard")

# User inputs for new prediction
probability = st.number_input("Enter Probability", min_value=0.0, max_value=1.0, value=0.5)
home_position = st.number_input("Enter Home Position", min_value=1, max_value=50, value=10)
away_position = st.number_input("Enter Away Position", min_value=1, max_value=50, value=20)
position_diff = abs(home_position - away_position)
odd = st.number_input("Enter ODD value", min_value=1.0, value=1.5)

outcome = st.selectbox("Select Outcome", label_encoders['Outcome'].classes_)
result = st.selectbox("Select Result", label_encoders['Result'].classes_)
difficulty = st.selectbox("Select Difficulty", label_encoders['Difficulty'].classes_)

# Prepare input data
new_data = pd.DataFrame({
    'Probability': [probability],
    'Home_Position': [home_position],
    'Away_Position': [away_position],
    'Position_Difference': [position_diff],
    'ODD': [odd],
    'Outcome': label_encoders['Outcome'].transform([outcome]),
    'Result': label_encoders['Result'].transform([result]),
    'Difficulty': label_encoders['Difficulty'].transform([difficulty])
})

# Scale data
new_data_scaled = scaler.transform(new_data)

# Predict button
if st.button("Predict Bet Safety"):
    prediction = rf_model.predict(new_data_scaled)
    prediction_label = label_encoders['bet'].inverse_transform(prediction)
    st.success(f"Predicted Bet Safety: {prediction_label[0]}")
