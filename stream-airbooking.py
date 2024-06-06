from io import StringIO
import logging
import joblib
import pandas as pd  
import requests
import streamlit as st 

# URL untuk file CSV
csv_url = 'https://github.com/hayhrmwn/tubes-data-sceince/blob/main/customer_booking.csv'
# URL untuk file model Python
model_url = 'https://github.com/hayhrmwn/tubes-data-sceince/raw/main/airlanes_booking_uas.joblib'

# Mengambil konten dari GitHub
airbook_model = requests.get(csv_url)
model_content = requests.get(model_url).content

# Memuat model menggunakan joblib
airlanes_model = joblib.load(BytesIO(model_content))

if airbook_model.status_code == 200:
    # Mengubah konten ke dalam DataFrame Pandas
    try:
        df = pd.read_csv(StringIO(airbook_model.text))
        # Tampilkan DataFrame
        st.write(df)
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file CSV: {e}")
else:
    st.error(f"Terjadi kesalahan saat mengambil file CSV: {airbook_model.status_code}")

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
            prediction = airlanes_model.predict([[sales_channel, trip_type, flight_day, route, booking_origin]])
            logging.info("Prediksi berhasil dilakukan.")

            if prediction[0] == 1:
                airbook_prediction = 'Perjalanan anda tepat'
            else:
                airbook_prediction = 'Perjalanan anda kurang tepat'
        except Exception as e:
            logging.error(f"Terjadi kesalahan saat prediksi: {e}")
            st.error(f"Terjadi kesalahan saat prediksi: {e}")

st.success(airbook_prediction)
