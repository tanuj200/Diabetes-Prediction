import streamlit as st
import joblib
import numpy as np
import os 

MODEL_PATH = os.path.join(os.path.dirname(__file__), "Logistic_regression.joblib")
model = joblib.load(MODEL_PATH)

st.title("Diabetes Prediction App")

pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose Level", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin Level", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0)


if st.button("Predict"):
    input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]).reshape(1, -1)
    prediction = model.predict(input_data)[0]
    
    result = "Diabetic" if prediction == 1 else "Not Diabetic"
    st.write(f"### Prediction: **{result}**")