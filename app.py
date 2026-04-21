import streamlit as st
import pandas as pd
from joblib import load

# Load dataset and models
df = pd.read_csv("dataset/final_model_data.csv")
rating_model = load("models/rating_model.joblib")
encoders = load("models/encoders.joblib")

st.title("Tourism Experience Rating Prediction")

# Select inputs
continent = st.selectbox("Continent", sorted(df["Continent"].unique()))
country = st.selectbox("Country", sorted(df["Country"].unique()))
attraction_type = st.selectbox("Attraction Type ID", sorted(df["AttractionTypeId"].unique()))

user_avg_rating = st.number_input("User Average Rating", min_value=0.0, max_value=5.0, value=4.0)
attraction_avg_rating = st.number_input("Attraction Average Rating", min_value=0.0, max_value=5.0, value=4.0)

# Predict button
if st.button("Predict Rating"):
    # Encode categorical inputs
    cont_encoded = encoders["Continent"].transform([continent])[0]
    country_encoded = encoders["Country"].transform([country])[0]
    type_encoded = encoders["AttractionTypeId"].transform([attraction_type])[0]

    input_df = pd.DataFrame([{
        
        "Continent": cont_encoded,
        "Country": country_encoded,
        "AttractionTypeId": type_encoded,
        "UserAvgRating": user_avg_rating,
        "AttractionAvgRating": attraction_avg_rating
    }])

    prediction = rating_model.predict(input_df)
    st.success(f"Predicted Rating: {prediction[0]:.2f}")
