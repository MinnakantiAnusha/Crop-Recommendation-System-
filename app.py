import streamlit as st
from PIL import Image
import numpy as np
import pickle

# Load the image
image = Image.open('crop.jpg')

# Load the crop recommendation model
model = pickle.load(open('crop_model.pkl', 'rb'))

# Define the feature names
feature_names = ['nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall']

# Create the Streamlit app
st.title('Crop Recommendation System')
st.image(image, caption='Crop Recommendation System', use_column_width=True)

# Create the form for user inputs
with st.form(key='crop_recommendation'):
    st.header('Enter the following details:')

    # User inputs
    nitrogen = st.number_input('Nitrogen (N)', min_value=0.0)
    phosphorus = st.number_input('Phosphorus (P)', min_value=0.0)
    potassium = st.number_input('Potassium (K)', min_value=0.0)
    temperature = st.number_input('Temperature', min_value=0.0)
    humidity = st.number_input('Humidity', min_value=0.0)
    ph = st.number_input('pH', min_value=0.0)
    rainfall = st.number_input('Rainfall', min_value=0.0)

    # Submit button
    submit = st.form_submit_button(label='Predict')

if submit:
    # Create the input array
    input_data = np.array([nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall])

    # Make the prediction
    prediction = model.predict([input_data])
    probabilities = model.predict_proba([input_data])

    # Display the results
    st.write('Crop Recommendation Results:')
    st.write('Top 3 Recommended Crops:')
    st.write(f'1. {prediction[0]} (Probability: {probabilities[0][0]:.2f})')
    st.write(f'2. {prediction[1]} (Probability: {probabilities[0][1]:.2f})')
    st.write(f'3. {prediction[2]} (Probability: {probabilities[0][2]:.2f})')
