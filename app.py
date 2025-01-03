import streamlit as st
import numpy as np
import pickle
import pandas as pd
import tensorflow as tf

# Load the model, label encoder, and scaler once at the start
model = tf.keras.models.load_model('crop.h5')

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('standard_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Function to read HTML files
def load_html(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# Load HTML content
index_html = load_html('index.html')
result_html_template = load_html('result.html')

# Display the index page
st.markdown(index_html, unsafe_allow_html=True)

# Collect user inputs
N = st.number_input("Nitrogen (N)", min_value=0.0)
P = st.number_input("Phosphorus (P)", min_value=0.0)
K = st.number_input("Potassium (K)", min_value=0.0)
temperature = st.number_input("Temperature", min_value=0.0)
humidity = st.number_input("Humidity", min_value=0.0)
ph = st.number_input("pH", min_value=0.0)
rainfall = st.number_input("Rainfall", min_value=0.0)

# Predict button
if st.button("Predict"):
    try:
        # Prepare features for prediction
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        features_scaled = scaler.transform(features)

        # Predict the crop
        prediction = model.predict(features_scaled)

        # Top 3 Recommendations
        top_3_indices = np.argsort(prediction[0])[-3:][::-1]
        top_3_crops = label_encoder.inverse_transform(top_3_indices)
        top_3_probs = prediction[0][top_3_indices]

        # Create result HTML
        result_html = result_html_template.replace("{{results}}", 
            "<br>".join([f"{crop} (Probability: {prob:.2f})" for crop, prob in zip(top_3_crops, top_3_probs)]))

        # Display results using HTML
        st.markdown(result_html, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Optional: Add a link to go back or reload
if st.button("Go Back"):
    st.experimental_rerun()
