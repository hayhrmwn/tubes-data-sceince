import streamlit as st
import pandas as pd
import requests
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Function to download a file from a URL
def download_file(url, local_path):
    response = requests.get(url)
    response.raise_for_status()  # Ensure we notice bad responses
    with open(local_path, 'wb') as file:
        file.write(response.content)

# Model URL and local path
model_url = 'https://github.com/hayhrmwn/tubes-data-sceince/raw/main/rf_model_wants_extra_baggage.pkl'
local_model_path = 'rf_model_wants_extra_baggage.pkl'

# CSV URL and local path
csv_url = 'https://github.com/hayhrmwn/tubes-data-sceince/raw/main/customer_booking.csv'
local_csv_path = 'customer_booking.csv'

# Download the model if it doesn't exist locally
if not os.path.exists(local_model_path):
    download_file(model_url, local_model_path)

# Download the CSV if it doesn't exist locally
if not os.path.exists(local_csv_path):
    download_file(csv_url, local_csv_path)

# Load original data from the local CSV file
df_original = pd.read_csv(local_csv_path, encoding='latin1')

# Streamlit interface
st.title("Customer Booking Prediction")

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

# Preprocess the original data to match the training data
df_original.drop(['booking_complete', 'flight_duration', 'route', 'wants_extra_baggage'], axis=1, inplace=True)  # Drop irrelevant columns and the target
encoder = OneHotEncoder(drop='first', handle_unknown='ignore')
encoded_cols = encoder.fit_transform(df_original[['booking_origin']])
encoded_df = pd.DataFrame(encoded_cols.toarray(), columns=encoder.get_feature_names_out(['booking_origin']))
df_original = pd.concat([df_original.select_dtypes(exclude='object'), encoded_df], axis=1)

# Load the model directly from URL
try:
    response = requests.get(model_url)
    response.raise_for_status()
    rf_model = joblib.load(response.content)
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Get feature names used during training (excluding the target)
feature_names_used_in_training = rf_model.feature_names_in_

# One-hot encode categorical features
encoded_input = encoder.transform(input_data[['booking_origin']]).toarray()
encoded_df = pd.DataFrame(encoded_input, columns=encoder.get_feature_names_out(['booking_origin']))
input_data = pd.concat([input_data.drop('booking_origin', axis=1), encoded_df], axis=1)

# Reorder columns to match training data
input_data = input_data[feature_names_used_in_training]

# Make prediction
try:
    prediction = rf_model.predict_proba(input_data)[0][1]
except Exception as e:
    st.error(f"Error making prediction: {e}")
    st.stop()

# Determine if wants extra baggage or not
wants_baggage = "wants extra baggage" if prediction > 0.5 else "doesn't want extra baggage"

# Display prediction result using st.success
st.success(f"Prediction: {prediction:.2f}\nThe customer {wants_baggage}")

if __name__ == '__main__':
    st._is_running_with_streamlit = True
    st.run()
