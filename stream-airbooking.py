import pickle
import pandas as pd  # type: ignore
import streamlit as st  # type: ignore

# Membaca data dari file CSV
airbooking_data = pd.read_csv('customer_booking.csv')

# Menampilkan beberapa baris pertama dari data
print(airbooking_data.head())

# Memuat model prediksi yang disimpan dalam file pickle
with open('model.pkl', 'rb') as model_file:
    airbooking_model = pickle.load(model_file)

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
    # Melakukan prediksi dengan model
    prediction = airbooking_model.predict([[sales_channel, trip_type, flight_day, route, booking_origin]])

    if prediction[0] == 1:
        airbook_prediction = 'Perjalanan anda tepat'
    else:
        airbook_prediction = 'Perjalanan anda kurang tepat'

st.success(airbook_prediction)
