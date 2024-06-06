from io import StringIO
import logging
import pickle
import pandas as pd  
import requests
import streamlit as st 

# Load the model
model_url = 'https://github.com/hayhrmwn/tubes-data-sceince/raw/main/airlines_booking_uas.pkl'
response = requests.get(model_url)

if response.status_code == 200:
    try:
        airbooking_model = pickle.loads(response.content)
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        st.error("Failed to load model.")
else:
    logging.error(f"Failed to download model. Status code: {response.status_code}")
    st.error("Failed to download model.")

# URL of the CSV file
csv_url = 'https://github.com/hayhrmwn/tubes-data-sceince/raw/main/customer_booking.csv'

csv_content = requests.get(csv_url).content

if csv_content:
    # Convert content to a DataFrame
    try:
        airbook_data = pd.read_csv(StringIO(csv_content.decode('utf-8')))
        # Display DataFrame
        st.write(airbook_data)
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
else:
    st.error("Error fetching CSV file.")

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
