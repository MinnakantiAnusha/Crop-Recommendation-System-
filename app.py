import streamlit as st
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import load_model
import tensorflow as tf

def main():
    st.set_page_config(page_title="Crop Recommendation System")

    # Load the model
    model = tf.keras.models.load_model('crop.h5')

    # Load the label encoder
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    # Load the standard scaler
    with open('standard_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # User input
    st.title("Crop Recommendation System")
    st.write("Please enter the following information:")

    N = st.number_input("Nitrogen (N):", min_value=0.0, step=1.0)
    P = st.number_input("Phosphorus (P):", min_value=0.0, step=1.0)
    K = st.number_input("Potassium (K):", min_value=0.0, step=1.0)
    temperature = st.number_input("Temperature (°C):", min_value=-50.0, step=0.1)
    humidity = st.number_input("Humidity (%):", min_value=0.0, step=0.1)
    ph = st.number_input("Soil pH:", min_value=0.0, step=0.01)
    rainfall = st.number_input("Rainfall (mm):", min_value=0.0, step=0.1)

    if st.button("Predict"):
        try:
            # Prepare input
            features = [N, P, K, temperature, humidity, ph, rainfall]
            features_scaled = scaler.transform([features])

            # Prediction
            prediction = model.predict(features_scaled)

            # Top 3 Recommendations
            top_3_indices = np.argsort(prediction[0])[-3:][::-1]
            top_3_crops = label_encoder.inverse_transform(top_3_indices)
            top_3_probs = prediction[0][top_3_indices]

            # Display results
            st.subheader("Top 3 Recommended Crops:")
            for crop, prob in zip(top_3_crops, top_3_probs):
                st.write(f"- {crop} (Probability: {prob:.2f})")
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
