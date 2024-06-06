import logging
import pickle
from urllib import request
import pandas as pd 
import streamlit as st 

csv_url = 'https://github.com/hayhrmwn/tubes-data-sceince/raw/main/customer_booking.csv'
airbook_model = 'https://github.com/hayhrmwn/tubes-data-sceince/blob/main/airlanes_booking_uas.py'

csv_content = request.get(csv_url).content
airbook_content = request.get(airbook_model).content

st.title ("Prediksi data analysis airbooking ")

col1, col2 = st.column(2)

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
        # Pastikan model sudah dimuat sebelumnya
        if not airbook_model:
            st.error("Model prediksi tidak tersedia.")
        else:
            try:
                # Lakukan prediksi dengan model (gunakan kode yang sesuai)
                prediction = airbook_model.predict([[sales_channel, trip_type, flight_day, route, booking_origin]])
                logging.info("Prediksi berhasil dilakukan.")

                if prediction[0] == 1:
                    airbook_prediction = 'Perjalanan anda tepat'
                else:
                    airbook_prediction = 'Perjalanan anda kurang tepat'
            except Exception as e:
                logging.error(f"Terjadi kesalahan saat prediksi: {e}")
                st.error(f"Terjadi kesalahan saat prediksi: {e}")

st.success(airbook_prediction)



