import streamlit as st
import joblib

# Load models
rain_model = joblib.load('rain_model.pkl')
rainfall_model = joblib.load('rainfall_model.pkl')
yield_model = joblib.load('yield_model.pkl')

st.title("🌾 AI Farmer Decision Support System")

temp = st.number_input("Annual Temperature (°C)", value=20.0)
wet_temp = st.number_input("Wettest Quarter Temp (°C)", value=15.0)
field_area = st.number_input("Field Area (ha)", value=1.0)

if st.button("Predict"):
    rain_pred = rain_model.predict([[temp, wet_temp]])[0]
    rainfall_pred = rainfall_model.predict([[temp, wet_temp]])[0]
    yield_pred = yield_model.predict([[field_area, temp]])[0]

    st.success(f"Rain Expected: {rain_pred}")
    st.info(f"Expected Rainfall: {rainfall_pred:.2f} mm")
    st.success(f"Predicted Yield: {yield_pred:.2f} kg")
