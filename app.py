import streamlit as st
import joblib
import pandas as pd

# Load models
rain_model = joblib.load('rain_model.pkl')
rainfall_model = joblib.load('rainfall_model.pkl')
yield_model = joblib.load('yield_model.pkl')

st.title("🌾 AI Farmer Decision Support System")

# Collect all inputs required by your CatBoost models
region = st.selectbox("Region", ["TIGRAY", "AMHARA", "OROMIA", "SNNPR"])
agro_zone = st.selectbox("Agro-Ecological Zone", ["Tropic-cool/humid", "Tropic-warm/semi-arid"])
field_area = st.number_input("Field Area (m²)", value=1000.0)
slope = st.number_input("Slope (%)", value=5.0)
irrigation = st.selectbox("Irrigation", ["YES", "NO"])
cropping_method = st.selectbox("Cropping Method", ["PURESTAND", "INTERCROPPING"])
fallow_status = st.selectbox("Fallow Status", ["YES", "NO"])
annual_rainfall = st.number_input("Annual Rainfall (mm)", value=800.0)
temp = st.number_input("Annual Temperature (°C)", value=20.0)
wet_temp = st.number_input("Wettest Quarter Temp (°C)", value=15.0)
latitude = st.number_input("Latitude", value=14.0)
longitude = st.number_input("Longitude", value=38.0)

if st.button("Predict"):
    # Construct input dataframe matching all model features
    input_df = pd.DataFrame([{
        'Region': region,
        'Agro-Ecological Zone': agro_zone,
        'Field Area': field_area,
        'Slope': slope,
        'Irrigation': irrigation,
        'Cropping Method': cropping_method,
        'Fallow Status': fallow_status,
        'Annual Rainfall': annual_rainfall,
        'Annual Mean Temperature': temp,
        'Wettest Quarter Temp': wet_temp,
        'Latitude': latitude,
        'Longitude': longitude
    }])

    rain_pred = rain_model.predict(input_df)[0]
    rainfall_pred = rainfall_model.predict(input_df)[0]
    yield_pred = yield_model.predict(input_df)[0]

    st.success(f"Rain Expected: {rain_pred}")
    st.info(f"Expected Rainfall: {rainfall_pred:.2f} mm")
    st.success(f"Predicted Yield: {yield_pred:.2f} kg")
