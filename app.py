import streamlit as st
import joblib
import pandas as pd

# Load models
rain_model = joblib.load('rain_model.pkl')
rainfall_model = joblib.load('rainfall_model.pkl')
yield_model = joblib.load('yield_model.pkl')

st.title("🌾 AI Farmer Decision Support System")

# Collect inputs
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
    # Master dictionary mapping UI input names
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

    def predict_model(model):
        # Extract exact expected feature names from CatBoost model
        try:
            expected_features = model.feature_names_
            df = pd.DataFrame([{col: data.get(col, 0) for col in expected_features}])
        except AttributeError:
            df = pd.DataFrame([data])
        return model.predict(df)[0]

    rain_pred = predict_model(rain_model)
    rainfall_pred = predict_model(rainfall_model)
    yield_pred = predict_model(yield_model)

    st.success(f"Rain Expected: {rain_pred}")
    st.info(f"Expected Rainfall: {rainfall_pred:.2f} mm")
    st.success(f"Predicted Yield: {yield_pred:.2f} kg")
