import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64

background_image_path = "red_wine.jpg"   # your background image

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return encoded

base64_image = get_base64_image(background_image_path)

page_bg = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/png;base64,{base64_image}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}

h1, h2, h3, h4, h5, h6, p, label, span, div {{
    color: #FFFFFF !important;   /* Pure white text */
}}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

st.title("🍷 Red Wine Quality Prediction")

model = joblib.load("final_rf_model.pkl")
scaler = joblib.load("scaler.pkl")
FEATURES = scaler.feature_names_in_

st.write("Enter the wine feature values below:")

inputs = {}
for col in FEATURES:
    inputs[col] = st.number_input(col, step=0.01, format="%.4f")

if st.button("Predict Quality"):
    input_df = pd.DataFrame([inputs], columns=FEATURES)
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    st.subheader(f"⭐ Predicted Wine Quality: {prediction}")
