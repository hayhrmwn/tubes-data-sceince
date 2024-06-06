from io import StringIO
import logging
import pickle
import pandas as pd  
import requests
import streamlit as st 

csv_url = 'https://github.com/hayhrmwn/tubes-data-sceince/raw/main/customer_booking.csv'

csv_content = requests.get(csv_url).content

if csv_content:
    # Try different encodings to read the CSV file
    encodings = ['utf-8', 'latin1']
    for encoding in encodings:
        try:
            airbook_data = pd.read_csv(StringIO(csv_content.decode(encoding)))
            # If successful, break the loop
            break
        except Exception as e:
            logging.warning(f"Failed to read CSV with encoding {encoding}: {e}")
            airbook_data = None
else:
    st.error("Failed to fetch CSV file.")

if airbook_data is None:
    st.error("Failed to read CSV file with all attempted encodings.")
else:
    st.write(airbook_data)

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
    else:
        # Pastikan model sudah dimuat sebelumnya
        if not airbooking_model:
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
