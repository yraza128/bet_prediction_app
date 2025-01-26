import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("bet_safety_model.pkl")

# App title
st.title("Bet Safety Prediction App")

# Sidebar for user inputs
st.sidebar.header("Input Game Details")

# User input fields
home_position = st.sidebar.number_input("Home Team Position", min_value=1, max_value=50, step=1, value=10)
away_position = st.sidebar.number_input("Away Team Position", min_value=1, max_value=50, step=1, value=20)
probability = st.sidebar.slider("Home Win Probability (%)", min_value=0, max_value=100, step=1, value=70) / 100
odd = st.sidebar.number_input("Home Win Odds", min_value=1.0, max_value=100.0, step=0.1, value=1.5)

# Calculate Position Difference
position_difference = abs(home_position - away_position)

# Display calculated position difference
st.write(f"Position Difference: {position_difference}")

# Predict button
if st.button("Predict Bet Safety"):
    # Prepare the input data
    input_data = pd.DataFrame({
        "Probability": [probability],
        "Home_Position": [home_position],
        "Away_Position": [away_position],
        "Position_Difference": [position_difference],
        "ODD": [odd],
        "Outcome": [0],  # Placeholder
        "Difficulty": [2],  # Placeholder (e.g., "Normal")
    })

    # Make prediction
    prediction = model.predict(input_data)[0]
    bet_safety = "Safe" if prediction == 1 else "Unsafe"

    # Display the result
    st.subheader(f"The bet is predicted to be: **{bet_safety}**")

# Footer
st.write("---")
st.write("This app uses a machine learning model to predict bet safety.")
