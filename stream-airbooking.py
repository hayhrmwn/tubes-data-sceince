import logging
import pickle
from urllib import request
import pandas as pd
import streamlit as st
import io  # Importing io for StringIO

# URL for the CSV and the model
csv_url = 'https://github.com/hayhrmwn/tubes-data-sceince/raw/main/customer_booking.csv'
airbook_model_url = 'https://github.com/hayhrmwn/tubes-data-sceince/raw/main/airlanes_booking_uas.py'

# Download CSV content
response = request.urlopen(csv_url)
csv_content = response.read()

# Load the CSV into a pandas DataFrame
data = pd.read_csv(io.StringIO(csv_content.decode('utf-8')))  # Use io.StringIO

# Download the model
response = request.urlopen(airbook_model_url)
model_content = response.read()

# Load the model using pickle
with open('/tmp/airbook_model.pkl', 'wb') as f:
    f.write(model_content)

with open('/tmp/airbook_model.pkl', 'rb') as f:
    airbook_model = pickle.load(f)

# Streamlit application
st.title("Prediksi data analysis airbooking")

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

if st.button('Tes Prediksi'):
    if any(not val for val in [sales_channel, trip_type, flight_day, route, booking_origin]):
        st.error("Semua input harus diisi.")
    else:
        try:
            # Ensure all inputs are of string type
            inputs = [sales_channel, trip_type, flight_day, route, booking_origin]
            # Perform prediction with the model
            prediction = airbook_model.predict([inputs])
            logging.info("Prediksi berhasil dilakukan.")

            if prediction[0] == 1:
                airbook_prediction = 'Perjalanan anda tepat'
            else:
                airbook_prediction = 'Perjalanan anda kurang tepat'
        except Exception as e:
            logging.error(f"Terjadi kesalahan saat prediksi: {e}")
            st.error(f"Terjadi kesalahan saat prediksi: {e}")

st.success(airbook_prediction)
