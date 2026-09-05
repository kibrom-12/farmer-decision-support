import gradio as gr
import joblib
import pandas as pd

# Load saved models
rain_model = joblib.load('rain_model.pkl')
rainfall_model = joblib.load('rainfall_model.pkl')
yield_model = joblib.load('yield_model.pkl')

def predict_farmer_outcomes(temp, wet_temp, field_area):
    # Model prediction logic
    rain_pred = rain_model.predict([[temp, wet_temp]])[0]
    rainfall_pred = rainfall_model.predict([[temp, wet_temp]])[0]
    yield_pred = yield_model.predict([[field_area, temp]])[0]
    
    return f"Rain Expected: {rain_pred}", f"Expected Rainfall: {rainfall_pred:.2f} mm", f"Predicted Yield: {yield_pred:.2f} kg"

demo = gr.Interface(
    fn=predict_farmer_outcomes,
    inputs=[
        gr.Number(label="Annual Temperature (°C)"),
        gr.Number(label="Wettest Quarter Temp (°C)"),
        gr.Number(label="Field Area (ha)")
    ],
    outputs=[
        gr.Textbox(label="Rain Prediction"),
        gr.Textbox(label="Rainfall Amount"),
        gr.Textbox(label="Crop Yield")
    ],
    title="Farmer Decision Support System"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=10000)
