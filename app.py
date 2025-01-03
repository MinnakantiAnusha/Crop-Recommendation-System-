import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import load_model
from jinja2 import Environment, FileSystemLoader

# Load the model
model = load_model('crop.h5')

# Load the label encoder
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load the standard scaler
with open('standard_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Set up Jinja2 environment
env = Environment(loader=FileSystemLoader('templates'))  # Assuming templates are in the 'templates' directory

def main():
    st.set_page_config(page_title="Crop Recommendation System")

    # Render the index.html template
    index_template = env.get_template('index.html')
    st.markdown(index_template.render(background_image='crop.jpg'), unsafe_allow_html=True)

    # Inputs
    N = st.number_input("Nitrogen (N):", min_value=0.0, step=1.0)
    P = st.number_input("Phosphorus (P):", min_value=0.0, step=1.0)
    # ... (rest of the input fields)

    # Predict button
    if st.button("Predict"):
        try:
            # Prepare input
            features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            features_scaled = scaler.transform(features)

            # Prediction
            prediction = model.predict(features_scaled)

            # Top 3 Recommendations
            top_3_indices = np.argsort(prediction[0])[-3:][::-1]
            top_3_crops = label_encoder.inverse_transform(top_3_indices)
            top_3_probs = prediction[0][top_3_indices]

            # Render the result.html template
            result_template = env.get_template('result.html')
            st.markdown(result_template.render(top_3_crops=top_3_crops, top_3_probs=top_3_probs, background_image='crop_result.jpg'), unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
