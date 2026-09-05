import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load models
rain_model = joblib.load('rain_model.pkl')
rainfall_model = joblib.load('rainfall_model.pkl')
yield_model = joblib.load('yield_model.pkl')

st.title("🌾 AI Farmer Decision Support System")

# Inputs
region = st.selectbox("Region", ["1. TIGRAY", "2. AMHARA", "3. OROMIA", "4. SNNPR"])
agro_zone = st.selectbox("Agro-Ecological Zone", ["Tropic-cool/humid", "Tropic-warm/semi-arid"])
field_area = st.number_input("Field Area (m²)", value=1000.0)
slope = st.number_input("Slope (%)", value=5.0)
irrigation = st.selectbox("Irrigation", ["1. YES", "2. NO"])
cropping_method = st.selectbox("Cropping Method", ["1. PURESTAND", "2. INTERCROPPING"])
fallow_status = st.selectbox("Fallow Status", ["1. YES", "2. NO"])
annual_rainfall = st.number_input("Annual Rainfall (mm)", value=800.0)
temp = st.number_input("Annual Mean Temperature (°C)", value=20.0)
wet_temp = st.number_input("Wettest Quarter Temp (°C)", value=15.0)
latitude = st.number_input("Latitude", value=14.0)
longitude = st.number_input("Longitude", value=38.0)

if st.button("Predict"):
    data = {
        'Region': region,
        'Agro-Ecological Zone': agro_zone,
        'Field Area (m²)': field_area,
        'Field Area': field_area,
        'Slope (%)': slope,
        'Slope': slope,
        'Irrigation': irrigation,
        'Cropping Method': cropping_method,
        'Fallow Status': fallow_status,
        'Annual Rainfall (mm)': annual_rainfall,
        'Annual Rainfall': annual_rainfall,
        'Annual Mean Temperature (°C)': temp,
        'Annual Mean Temperature': temp,
        'Wettest Quarter Temp (°C)': wet_temp,
        'Wettest Quarter Temp': wet_temp,
        'Latitude': latitude,
        'Longitude': longitude
    }

    def run_prediction(model_obj):
        # Extract prediction cleanly
        if hasattr(model_obj, 'feature_names_'):
            cols = model_obj.feature_names_
            df = pd.DataFrame([{c: data.get(c, 0) for c in cols}])
        else:
            df = pd.DataFrame([data])
        
        res = model_obj.predict(df)
        if isinstance(res, (list, np.ndarray, pd.Series)):
            return res[0]
        return res

    rain_pred = run_prediction(rain_model)
    rainfall_pred = run_prediction(rainfall_model)
    yield_pred = run_prediction(yield_model)

    st.success(f"Rain Expected: {rain_pred}")
    st.info(f"Expected Rainfall: {float(rainfall_pred):.2f} mm")
    st.success(f"Predicted Yield: {float(yield_pred):.2f} kg")
