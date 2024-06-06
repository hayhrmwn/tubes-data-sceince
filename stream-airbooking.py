import logging
import pickle
from urllib import request
import pandas as pd
import streamlit as st
import os
import xgboost as xgb  # Pastikan xgboost sudah terinstal
from sklearn.model_selection import train_test_split, cross_val_score # type: ignore
from sklearn.metrics import accuracy_score # type: ignore


# URL untuk CSV dan Model
csv_url = 'https://github.com/hayhrmwn/tubes-data-sceince/raw/main/customer_booking.csv'
model_url = 'https://github.com/hayhrmwn/tubes-data-sceince/raw/main/airlanes_booking_uas.pkl'

# Download dan baca data CSV
csv_response = request.urlopen(csv_url)
df = pd.read_csv(csv_response)

# Download dan load model XGBoost
model_path = 'airlanes_booking_uas.pkl'
if not os.path.exists(model_path):
    model_response = request.urlopen(model_url)
    with open(model_path, 'wb') as f:
        f.write(model_response.read())

with open(model_path, 'rb') as f:
    model = pickle.load(f)

st.title("Prediksi Data Analysis Airbooking")

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
            # Lakukan prediksi dengan model
            prediction = model.predict(pd.DataFrame([[sales_channel, trip_type, flight_day, route, booking_origin]],
                                                    columns=['Sales_Channel', 'Trip_Type', 'Flight_Day', 'Route', 'Booking_Origin']))
            logging.info("Prediksi berhasil dilakukan.")

            if prediction[0] == 1:
                airbook_prediction = 'Perjalanan anda tepat'
            else:
                airbook_prediction = 'Perjalanan anda kurang tepat'
        except Exception as e:
            logging.error(f"Terjadi kesalahan saat prediksi: {e}")
            st.error(f"Terjadi kesalahan saat prediksi: {e}")

st.success(airbook_prediction)
