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

# Rest of your code for prediction...
