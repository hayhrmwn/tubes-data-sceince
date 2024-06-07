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
