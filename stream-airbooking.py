import logging
import pickle
import pandas as pd  
import streamlit as st 
import wget
import os

# Download the model
model_url = 'https://github.com/hayhrmwn/tubes-data-sceince/raw/main/airlines_booking_uas.pkl'
model_path = 'airlines_booking_uas.pkl'

if not os.path.exists(model_path):
    try:
        logging.info("Downloading model...")
        wget.download(model_url, model_path)
        logging.info("Model downloaded successfully.")
    except Exception as e:
        logging.error(f"Failed to download model: {e}")
        st.error("Failed to download model.")

# Load the model
try:
    with open(model_path, 'rb') as model_file:
        airbooking_model = pickle.load(model_file)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    st.error("Failed to load model.")

# URL of the CSV file
csv_url = 'https://github.com/hayhrmwn/tubes-data-sceince/raw/main/customer_booking.csv'

# Download the CSV file
try:
    airbook_data = pd.read_csv(csv_url)
    logging.info("CSV file loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load CSV file: {e}")
    st.error("Failed to load CSV file.")

st.title('Airbooking Prediction Model')

# Input columns
col1, col2 = st.columns(2)

with col1:
    sales_channel = st.text_input('Input Sales Channel')

with col2:
    trip_type = st.text_input('Input Trip Type')

with col1:
    flight_day = st.text_input('Input Flight Day')

with col2:
    route = st.text_input('Input Route')

with col1:
    booking_origin = st.text_input('Input Booking Origin')

airbook_prediction = ''

if st.button('Test Prediction'):
    if any(not val for val in [sales_channel, trip_type, flight_day, route, booking_origin]):
        st.error("All inputs must be filled.")
    else:
        # Ensure the model is loaded
        if not airbooking_model:
            st.error("Prediction model not available.")
        else:
            try:
                # Make prediction with the model
                prediction = airbooking_model.predict([[sales_channel, trip_type, flight_day, route, booking_origin]])
                logging.info("Prediction successful.")

                if prediction[0] == 1:
                    airbook_prediction = 'Your journey is appropriate'
                else:
                    airbook_prediction = 'Your journey is less appropriate'
            except Exception as e:
                logging.error(f"Error during prediction: {e}")
                st.error(f"Error during prediction: {e}")

st.success(airbook_prediction)
