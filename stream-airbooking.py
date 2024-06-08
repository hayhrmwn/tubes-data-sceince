import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import joblib
import requests
from io import BytesIO

# Function to download a file from a URL
def download_file(url):
    response = requests.get(url)
    response.raise_for_status()  # Ensure we notice bad responses
    return BytesIO(response.content)

# Load the pre-trained model
model_url = 'https://github.com/hayhrmwn/tubes-data-sceince/raw/main/rf_model_wants_extra_baggage.pkl'
rf_model = joblib.load(download_file(model_url))

encoder_url = 'https://github.com/hayhrmwn/tubes-data-sceince/raw/main/encoder_wants_extra_baggage.pkl'
encoder = joblib.load(download_file(encoder_url))

feature_names_url = 'https://github.com/hayhrmwn/tubes-data-sceince/raw/main/feature_names_wants_extra_baggage.pkl'
feature_names_used_in_training = joblib.load(download_file(feature_names_url))

# CSV URL
csv_url = 'https://github.com/hayhrmwn/tubes-data-sceince/raw/main/customer_booking.csv'

# Streamlit interface
st.title("Customer Booking Prediction")

# Load original data from the CSV file
@st.cache
def load_data():
    return pd.read_csv(csv_url, encoding='latin1')

df_original = load_data()

# User input
num_passengers = st.number_input('Number of Passengers', min_value=1, step=1)
sales_channel = st.selectbox('Sales Channel', ['Online', 'Offline'])
trip_type = st.selectbox('Trip Type', ['One Way', 'Round Trip'])
purchase_lead = st.number_input('Purchase Lead Time', min_value=0, step=1)
length_of_stay = st.number_input('Length of Stay', min_value=0, step=1)
flight_hour = st.number_input('Flight Hour', min_value=0, max_value=23, step=1)
flight_day = st.selectbox('Flight Day', ['Weekday', 'Weekend'])
route = st.text_input('Route')
booking_origin = st.selectbox('Booking Origin', df_original['booking_origin'].unique())
wants_preferred_seat = st.selectbox('Wants Preferred Seat', ['Yes', 'No'])
wants_in_flight_meals = st.selectbox('Wants In-flight Meals', ['Yes', 'No'])

# Create DataFrame from user input
input_data = pd.DataFrame({
    'num_passengers': [num_passengers],
    'sales_channel': [sales_channel],
    'trip_type': [trip_type],
    'purchase_lead': [purchase_lead],
    'length_of_stay': [length_of_stay],
    'flight_hour': [flight_hour],
    'flight_day': [flight_day],
    'route': [route],
    'booking_origin': [booking_origin],
    'wants_preferred_seat': [1 if wants_preferred_seat == 'Yes' else 0],
    'wants_in_flight_meals': [1 if wants_in_flight_meals == 'Yes' else 0]
})

# One-hot encode categorical features
encoded_input = encoder.transform(input_data[['booking_origin']]).toarray()
encoded_df = pd.DataFrame(encoded_input, columns=encoder.get_feature_names_out(['booking_origin']))
input_data = pd.concat([input_data.drop('booking_origin', axis=1), encoded_df], axis=1)

# Reorder columns to match training data
input_data = input_data[feature_names_used_in_training]

# Make prediction
prediction = rf_model.predict_proba(input_data)[0][1]

# Determine if wants extra baggage or not
wants_baggage = "wants extra baggage" if prediction > 0.5 else "doesn't want extra baggage"

# Display prediction result using st.success
st.success(f"Prediction: {prediction:.2f}\nThe customer {wants_baggage}")

