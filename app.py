import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import streamlit as st
import altair as alt

# Load and prepare data
@st.cache_data
def load_data():
    data = pd.read_csv('Extended_Crop_recommendation_4000.csv')
    return data

data = load_data()
X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = data['label']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Build and train the model
@st.cache_resource
def build_model():
    model = Sequential([
        Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(len(np.unique(y_encoded)), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_model()

@st.cache_resource
def train_model():
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=20, batch_size=4, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=0)
    return model

model = train_model()

# Streamlit UI
st.markdown("""
    <style>
        .header {
            font-size: 36px;
            font-weight: bold;
            color: #4CAF50; /* Green color for farming theme */
            text-align: center;
            padding: 10px;
            border-radius: 15px;
            background-color: #F9FBE7; /* Light green background */
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            margin-bottom: 20px;
        }
        .section-header {
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
            margin-bottom: 20px;
        }
        .sidebar {
            background-color: #E8F5E9; /* Light green for sidebar */
        }
        .plant-image {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100px;
            height: auto;
            padding-bottom: 10px;
        }
    </style>
    <div class="header">
        <span style="font-size: 36px;">🌾 GROW 🌾</span><br>
        Guidance for Recommended Optimal Yields
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar">', unsafe_allow_html=True)
st.sidebar.header('User Input Parameters')
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# User input method
input_method = st.sidebar.radio("Choose input method:", ("Sliders", "Text Boxes"))

def user_input_features():
    if input_method == "Sliders":
        N = st.sidebar.slider('Nitrogen', 0, 200, 100)
        P = st.sidebar.slider('Phosphorus', 0, 200, 50)
        K = st.sidebar.slider('Potassium', 0, 200, 100)
        temperature = st.sidebar.slider('Temperature', 10.0, 50.0, 25.0)
        humidity = st.sidebar.slider('Humidity', 20.0, 100.0, 50.0)
        ph = st.sidebar.slider('pH', 3.0, 10.0, 6.5)
        rainfall = st.sidebar.slider('Rainfall', 0.0, 300.0, 100.0)
    else:
        N = st.sidebar.number_input('Nitrogen', 0, 200, 100)
        P = st.sidebar.number_input('Phosphorus', 0, 200, 50)
        K = st.sidebar.number_input('Potassium', 0, 200, 100)
        temperature = st.sidebar.number_input('Temperature', 10.0, 50.0, 25.0)
        humidity = st.sidebar.number_input('Humidity', 20.0, 100.0, 50.0)
        ph = st.sidebar.number_input('pH', 3.0, 10.0, 6.5)
        rainfall = st.sidebar.number_input('Rainfall', 0.0, 300.0, 100.0)
    return np.array([N, P, K, temperature, humidity, ph, rainfall]).reshape(1, -1)

user_input = user_input_features()
user_input_scaled = scaler.transform(user_input)

# Predict the crop
prediction = model.predict(user_input_scaled)
top_3_indices = np.argsort(prediction[0])[-3:][::-1]
top_3_crops = label_encoder.inverse_transform(top_3_indices)
top_3_scores = prediction[0][top_3_indices]
top_3_probs = np.exp(top_3_scores) / np.sum(np.exp(top_3_scores))

# Create a DataFrame for plotting
df_top_crops = pd.DataFrame({
    'Crop': top_3_crops,
    'Probability': top_3_probs
})

# Plot using Altair
chart = alt.Chart(df_top_crops).mark_bar().encode(
    x=alt.X('Crop', title='Crop'),
    y=alt.Y('Probability', title='Probability'),
    color=alt.Color('Probability', scale=alt.Scale(scheme='blues')),  # Changed to blue color scheme
    tooltip=['Crop', 'Probability']
).properties(
    title='Top 3 Recommended Crops'
)

# Streamlit components
st.subheader('Top 3 Recommended Crops:')
for crop, prob in zip(top_3_crops, top_3_probs):
    st.write(f"{crop} (Probability: {prob:.2f})")

st.altair_chart(chart, use_container_width=True)
