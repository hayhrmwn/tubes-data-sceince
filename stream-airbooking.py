import logging
import pickle
import pandas as pd  
import streamlit as st 
import os

drive.mount('/content/drive')
data_path = '/content/drive/MyDrive/Dataset/customer_booking.csv'


try:
    airbooking_model = pd.read_csv(data_path)
    st.success("File CSV berhasil dibaca!")
    # Lakukan proses lain dengan data CSV di sini
except Exception as e:
    st.error(f"Terjadi kesalahan saat membaca file CSV: {e}")

st.title('Prediksi Model Airbooking')

# Membuat kolom input
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
    elif airbook_model is None:
        st.error("Model prediksi tidak tersedia.")
    else:
        try:
            # Melakukan prediksi dengan model
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
