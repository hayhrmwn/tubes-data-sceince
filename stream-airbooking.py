import os
import pickle
import pandas as pd  # type: ignore
import streamlit as st  # type: ignore
import logging

logging.basicConfig(level=logging.INFO)

# Jalur file
csv_path = 'customer_booking.csv'
model_path = 'model.pkl'

# Memastikan file CSV dan model ada
airbooking_data = None
airbooking_model = None

# Memeriksa keberadaan file CSV
if not os.path.exists(csv_path):
    logging.error(f"File CSV tidak ditemukan: {csv_path}")
    st.error(f"File CSV tidak ditemukan: {csv_path}")
else:
    # Membaca data dari file CSV dengan encoding yang berbeda jika utf-8 gagal
    try:
        airbooking_data = pd.read_csv(csv_path, encoding='utf-8')
        logging.info("File CSV dibaca dengan sukses dengan encoding utf-8.")
    except UnicodeDecodeError as e:
        logging.warning(f"Kesalahan decoding utf-8: {e}")
        try:
            airbooking_data = pd.read_csv(csv_path, encoding='latin1')
            logging.info("File CSV dibaca dengan sukses dengan encoding latin1.")
        except Exception as e:
            logging.error(f"Terjadi kesalahan saat membaca file CSV: {e}")
            st.error(f"Terjadi kesalahan saat membaca file CSV: {e}")

# Memeriksa keberadaan file model
if not os.path.exists(model_path):
    logging.error(f"File model tidak ditemukan: {model_path}")
    st.error(f"File model tidak ditemukan: {model_path}")
else:
    # Memuat model prediksi yang disimpan dalam file pickle
    try:
        with open(model_path, 'rb') as model_file:
            airbooking_model = pickle.load(model_file)
            logging.info(f"Model dimuat dengan sukses dari {model_path}.")
    except Exception as e:
        logging.error(f"Terjadi kesalahan saat memuat model: {e}")
        st.error(f"Terjadi kesalahan saat memuat model: {e}")

st.title('Prediksi Airbooking Analysis')

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
    elif airbooking_model is None:
        st.error("Model prediksi tidak tersedia.")
    else:
        try:
            # Melakukan prediksi dengan model
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
