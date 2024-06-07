import streamlit as st
import pandas as pd
import requests
import logging
import chardet
from io import BytesIO
import pickle
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import AdaBoostClassifier

st.title('Prediksi Model Airbooking')

csv_url = 'https://github.com/hayhrmwn/tubes-data-sceince/raw/main/customer_booking.csv'
xgb_model_url = 'https://github.com/hayhrmwn/tubes-data-sceince/raw/main/xgb_model.pkl'
lgb_model_url = 'https://github.com/hayhrmwn/tubes-data-sceince/raw/main/lgb_model.pkl'
ada_model_url = 'https://github.com/hayhrmwn/tubes-data-sceince/raw/main/ada_model.pkl'

def detect_encoding(url):
    response = requests.get(url)
    rawdata = response.content
    result = chardet.detect(rawdata)
    return result['encoding'], rawdata

def load_model(model_url):
    try:
        model_response = requests.get(model_url)
        model_response.raise_for_status()  # Raise an exception for HTTP errors
        with open('model.pkl', 'wb') as f:
            f.write(model_response.content)
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
        logging.info("Model prediksi berhasil dimuat.")
        return model
    except Exception as e:
        st.error(f"Gagal memuat model prediksi: {e}")
        return None

def preprocess_input(data):
    # Mengonversi kolom object menjadi kategori
    for column in data.columns:
        data[column] = data[column].astype('category')
    return data

def predict_airbooking(xgb_model, lgb_model, ada_model, sales_channel, trip_type, flight_day, route, booking_origin):
    try:
        # Lakukan prediksi dengan model XGBoost
        xgb_prediction = xgb_model.predict([[sales_channel, trip_type, flight_day, route, booking_origin]])[0]

        # Lakukan prediksi dengan model LightGBM
        lgb_prediction = lgb_model.predict([[sales_channel, trip_type, flight_day, route, booking_origin]])[0]

        # Lakukan prediksi dengan model AdaBoost
        ada_prediction = ada_model.predict([[sales_channel, trip_type, flight_day, route, booking_origin]])[0]

        # Gabungkan prediksi dari semua model
        combined_prediction = (xgb_prediction + lgb_prediction + ada_prediction) / 3

        if combined_prediction >= 0.5:
            return 'Perjalanan anda tepat'
        else:
            return 'Perjalanan anda kurang tepat'

def input_form():
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

    return sales_channel, trip_type, flight_day, route, booking_origin

def predict_button():
    if st.button('Tes Prediksi'):
        return True
    return False

encoding, rawdata = detect_encoding(csv_url)

if encoding:
    try:
        airbook_data = pd.read_csv(BytesIO(rawdata), encoding=encoding)
    except Exception as e:
        st.error(f"Gagal membaca file CSV dengan encoding {encoding}: {e}")
        airbook_data = None
else:
    st.error("Gagal mendeteksi encoding file CSV.")
    airbook_data = None

if airbook_data is not None:
    st.write(airbook_data)
else:
    st.error("Gagal membaca file CSV.")

xgb_model = load_model(xgb_model_url)
lgb_model = load_model(lgb_model_url)
ada_model = load_model(ada_model_url)

sales_channel, trip_type, flight_day, route, booking_origin = input_form()

if predict_button():
    if any(not val for val in [sales_channel, trip_type, flight_day, route, booking_origin]):
        st.error("Semua input harus diisi.")
    else:
        try:
            input_data = pd.DataFrame([[sales_channel, trip_type, flight_day, route, booking_origin]], 
                                      columns=['Sales Channel', 'Trip Type', 'Flight Day', 'Route', 'Booking Origin'])
            input_data = preprocess_input(input_data)
            logging.info(f"Input data for prediction: {input_data}")

            airbook_prediction = predict_airbooking(xgb_model, lgb_model, ada_model, sales_channel, trip_type, flight_day, route, booking_origin)

        except Exception as e:
            logging.error(f"Terjadi kesalahan saat prediksi: {e}")
            st.error(f"Terjadi kesalahan saat prediksi: {e}")

st.success(airbook_prediction)
