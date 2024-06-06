import pickle
import streamlit as st

airbooking_model = pickle.load(open('customer_booking.csv', 'rb'))

st.title('Prediksi Airbooking anlysis')

col1, col2 = st.columnns(2)

with col1 :
    sales_channel = st.text_input ('Input Sales Channel')

with col2 :
    Trip_type = st.text_input ('Input Trip Type')

with col1 :
    flight_day = st.text_input ('Input Flight Day')

with col2 :
    route = st.text_input ('Input Route')

with col1 :
    booking_origin = st.text_input ('Input Booking Origin')

airbook_predict = ''

if st.button('Tes prediksi '):
        airbook_prediction = airbooking_model.predict([[sales_channel,Trip_type,flight_day,route,booking_origin]])

        if(airbook_prediction[0] == 1):
             airbook_prediction = 'Perjalanan anda tepat'
        else :
             airbook_prediction = 'Perjalanan anda kurang tepat'

st.success(airbook_predict)