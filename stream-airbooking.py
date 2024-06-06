import logging
import pickle
from urllib import request
import pandas as pd
import streamlit as st

# URL for the CSV and the model
csv_url = 'https://github.com/hayhrmwn/tubes-data-sceince/raw/main/customer_booking.csv'
airbook_model_url = 'https://github.com/hayhrmwn/tubes-data-sceince/raw/main/airlanes_booking_uas.py'

# Download CSV content
response = request.urlopen(csv_url)
csv_content = response.read().decode('utf-8')

# Load the CSV into a pandas DataFrame
data = pd.read_csv(pd.compat.StringIO(csv_content))

# Download the model
response = request.urlopen(airbook_model_url)
model_content = response.read()

# Load the model using pickle
with open('/tmp/airbook_model.pkl', 'wb') as f:
    f.write(model_content)

with open('/tmp/airbook_model.pkl', 'rb') as f:
    airbook_model = pickle.load(f)

# Streamlit application
st.title("Prediksi data analysis airbooking")

col1, col2 = st.columns(2)

with col1:
    sales_channel = st.text_input('Input Sales Channel')

with col2:
    trip_type = st.text_input('Input Trip Type')

with col1:
    flight
