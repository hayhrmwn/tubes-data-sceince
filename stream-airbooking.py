import streamlit as st
import pandas as pd
import requests
from io import StringIO
import logging

st.title('Prediksi Model Airbooking')

csv_url = 'https://github.com/hayhrmwn/tubes-data-sceince/raw/main/customer_booking.csv'

# Function to read CSV with different encodings
def read_csv_with_encodings(url, encodings):
    for encoding in encodings:
        try:
            csv_content = requests.get(url).content
            df = pd.read_csv(StringIO(csv_content.decode(encoding)))
            return df
        except Exception as e:
            logging.warning(f"Gagal membaca CSV dengan enkoding {encoding}: {e}")
    return None

# List of encodings to try
encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
airbook_data = read_csv_with_encodings(csv_url, encodings)

if airbook_data is None:
    st.error("Gagal membaca file CSV dengan semua enkoding yang dicoba.")
else:
    # Menampilkan data CSV jika berhasil dibaca
    st.write(airbook_data)

# Inisialisasi model prediksi (gunakan kode yang sesuai)
airbooking_model = None  # Replace with actual model loading code

# Input kolom
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
        # Pastikan model sudah dimuat sebelumnya
        if not airbooking_model:
            st.error("Model prediksi tidak tersedia.")
        else:
            try:
                # Lakukan prediksi dengan model (gunakan kode yang sesuai)
                prediction = airbooking_model.predict([[sales_channel, trip_type, flight_day, route, booking_origin]])
                logging.info("Prediksi berhasil dilakukan.")

                if prediction[0] == 1:
                    airbook_prediction = 'Perjalanan anda tepat'
                else:
                    airbook_prediction = 'Perjalanan anda kurang tepat'
            except Exception as e:
                logging.error(f"Terjadi kesalahan saat prediksi: {e}")
                st.error(f"Terjadi kesalahan saat prediksi: {e}")

st.success(airbook_prediction)
