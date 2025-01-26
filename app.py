import streamlit as st
import pandas as pd
import joblib

# Load saved model and scaler
rf_model = joblib.load('bet_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Bet Safety Prediction Dashboard")

# User inputs for prediction
probability = st.number_input("Enter Probability", min_value=0.0, max_value=1.0, value=0.5)
home_position = st.number_input("Enter Home Position", min_value=1, max_value=50, value=10)
away_position = st.number_input("Enter Away Position", min_value=1, max_value=50, value=20)
position_diff = abs(home_position - away_position)
odd = st.number_input("Enter ODD value", min_value=1.0, value=1.5)

# Prepare input data (only the features used for prediction)
new_data = pd.DataFrame({
    'Probability': [probability],
    'Home_Position': [home_position],
    'Away_Position': [away_position],
    'Position_Difference': [position_diff],
    'ODD': [odd]
})

# Scale data
new_data_scaled = scaler.transform(new_data)

# Predict button
if st.button("Predict Bet Safety"):
    prediction = rf_model.predict(new_data_scaled)
    prediction_label = "Safe" if prediction[0] == 1 else "Unsafe"
    st.success(f"Predicted Bet Safety: {prediction_label}")
