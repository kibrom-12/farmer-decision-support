import streamlit as st
import joblib
import pandas as pd

# Load models
rain_model = joblib.load('rain_model.pkl')
rainfall_model = joblib.load('rainfall_model.pkl')
yield_model = joblib.load('yield_model.pkl')

st.title("🌾 AI Farmer Decision Support System")

# Collect inputs
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
    # Construct input dataframe
    input_df = pd.DataFrame([{
        'Region': str(region),
        'Agro-Ecological Zone': str(agro_zone),
        'Field Area': float(field_area),
        'Slope': float(slope),
        'Irrigation': str(irrigation),
        'Cropping Method': str(cropping_method),
        'Fallow Status': str(fallow_status),
        'Annual Rainfall': float(annual_rainfall),
        'Annual Mean Temperature': float(temp),
        'Wettest Quarter Temp': float(wet_temp),
        'Latitude': float(latitude),
        'Longitude': float(longitude)
    }])

    # Convert text columns to categorical type for CatBoost
    cat_cols = ['Region', 'Agro-Ecological Zone', 'Irrigation', 'Cropping Method', 'Fallow Status']
    for col in cat_cols:
        input_df[col] = input_df[col].astype('category')

    # Generate predictions
    rain_pred = rain_model.predict(input_df)[0]
    rainfall_pred = rainfall_model.predict(input_df)[0]
    yield_pred = yield_model.predict(input_df)[0]

    st.success(f"Rain Expected: {rain_pred}")
    st.info(f"Expected Rainfall: {rainfall_pred:.2f} mm")
    st.success(f"Predicted Yield: {yield_pred:.2f} kg")
