import streamlit as st
import pandas as pd
import requests
import logging
import chardet
from io import BytesIO
import pickle
import xgboost as xgb

st.title('Prediksi Model Airbooking')

# URL untuk data dan model
csv_url = 'https://github.com/hayhrmwn/tubes-data-sceince/raw/main/customer_booking.csv'
model_url = 'https://github.com/hayhrmwn/tubes-data-sceince/raw/main/xgb_model.pkl'

# Fungsi untuk mendeteksi encoding file
def detect_encoding(url):
    response = requests.get(url)
    rawdata = response.content
    result = chardet.detect(rawdata)
    return result['encoding'], rawdata

# Mendeteksi encoding dan membaca data
encoding, rawdata = detect_encoding(csv_url)

if encoding:
    try:
        airbook_data = pd.read_csv(BytesIO(rawdata), encoding=encoding)
        st.write(airbook_data)
    except Exception as e:
        st.error(f"Gagal membaca file CSV dengan encoding {encoding}: {e}")
else:
    st.error("Gagal mendeteksi encoding file CSV.")
    airbook_data = None

# Mengunduh dan memuat model prediksi dari URL
try:
    model_response = requests.get(model_url)
    model_response.raise_for_status()
    with open('xgb_model.pkl', 'wb') as f:
        f.write(model_response.content)
    with open('xgb_model.pkl', 'rb') as file:
        airbooking_model = pickle.load(file)
    logging.info("Model prediksi berhasil dimuat.")
except Exception as e:
    st.error(f"Gagal memuat model prediksi: {e}")
    airbooking_model = None

# Input kolom dari pengguna
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

# Fungsi untuk pra-pemrosesan input
def preprocess_input(data):
    # Mengonversi kolom object menjadi kategori
    for column in data.columns:
        data[column] = data[column].astype('category')
    return data

# Tombol untuk melakukan prediksi
if st.button('Tes Prediksi'):
    if any(not val for val in [sales_channel, trip_type, flight_day, route, booking_origin]):
        st.error("Semua input harus diisi.")
    else:
        if not airbooking_model:
            st.error("Model prediksi tidak tersedia.")
        else:
            try:
                # Membuat DataFrame dari input pengguna
                input_data = pd.DataFrame([[sales_channel, trip_type, flight_day, route, booking_origin]], 
                                          columns=['Sales Channel', 'Trip Type', 'Flight Day', 'Route', 'Booking Origin'])
                input_data = preprocess_input(input_data)
                logging.info(f"Input data for prediction: {input_data}")

                # Melakukan prediksi
                prediction = airbooking_model.predict(input_data)
                logging.info("Prediksi berhasil dilakukan.")

                if prediction[0] == 1:
                    airbook_prediction = 'Perjalanan anda tepat'
                else:
                    airbook_prediction = 'Perjalanan anda kurang tepat'
            except Exception as e:
                logging.error(f"Terjadi kesalahan saat prediksi: {e}")
                st.error(f"Terjadi kesalahan saat prediksi: {e}")

st.success(airbook_prediction)
