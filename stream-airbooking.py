from io import StringIO
import logging
import pickle
import pandas as pd  
import requests
import streamlit as st 
import os
import importlib.util

# Fungsi untuk mengunduh dan mengimpor modul airlines_booking_uas.py
def import_airlines_booking_model(module_url):
    # Direktori tempat menyimpan file sementara
    temp_dir = 'temp'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Path untuk menyimpan file sementara
    temp_file_path = os.path.join(temp_dir, 'airlines_booking_uas.py')

    # Mengunduh konten modul dari GitHub
    response = requests.get(module_url)

    if response.status_code == 200:
        # Menyimpan konten ke dalam file sementara
        with open(temp_file_path, 'wb') as f:
            f.write(response.content)

        # Mengimpor modul dari file sementara
        spec = importlib.util.spec_from_file_location("airlines_booking_uas", temp_file_path)
        airlines_booking_uas = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(airlines_booking_uas)

        # Mengembalikan model prediksi dari modul
        return airlines_booking_uas.airbooking_model
    else:
        print(f"Failed to download module from {module_url}. Status code: {response.status_code}")
        return None

# URL modul airlines_booking_uas.py di GitHub
module_url = 'https://github.com/hayhrmwn/tubes-data-sceince/raw/main/airlines_booking_uas.py'

# Impor model prediksi
airbooking_model = import_airlines_booking_model(module_url)

# Jika model berhasil diimpor, lanjutkan dengan aplikasi Streamlit
if airbooking_model:
    # URL file CSV
    csv_url = 'https://github.com/hayhrmwn/tubes-data-sceince/raw/main/customer_booking.csv'

    # Membaca isi file CSV dari URL
    csv_content = requests.get(csv_url).content

    if csv_content:
        # Mencoba beberapa enkoding yang berbeda untuk membaca file CSV
        encodings = ['utf-8', 'latin1']
        for encoding in encodings:
            try:
                # Membaca file CSV menggunakan enkoding yang sesuai
                airbook_data = pd.read_csv(StringIO(csv_content.decode(encoding)))
                # Jika berhasil, hentikan loop
                break
            except Exception as e:
                logging.warning(f"Gagal membaca CSV dengan enkoding {encoding}: {e}")
                airbook_data = None
    else:
        st.error("Gagal mengambil file CSV.")

    if airbook_data is None:
        st.error("Gagal membaca file CSV dengan semua enkoding yang dicoba.")
    else:
        # Menampilkan data CSV jika berhasil dibaca
        st.write(airbook_data)

    st.title('Prediksi Model Airbooking')

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
            try:
                # Lakukan prediksi dengan model
                prediction = airbooking_model.predict([[sales_channel, trip_type, flight_day, route, booking_origin]])
                logging.info("Predik
