import streamlit as st
import pickle

st.title("Predicting Diabetes")

# Input fields with unique labels
Pregnancies = st.number_input('Pregnancies', min_value=0, max_value=17)
Glucose = st.number_input('Glucose', min_value=0, max_value=200)
BloodPressure = st.number_input('BloodPressure', min_value=0, max_value=122)
SkinThickness = st.number_input('SkinThickness', min_value=0, max_value=99)
Insulin = st.number_input('Insulin', min_value=0, max_value=846)
BMI = st.number_input('BMI', min_value=0.0, max_value=67.1, value=1.0)
DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.000, max_value=2.420, value=1.0)
Age = st.number_input('Age', min_value=0, max_value=100)

# Load the model
with open('tree_cl.pkl', 'rb') as file:
    model = pickle.load(file)

# Predict only when the user clicks the button
if st.button('Predict'):
    output = model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    st.write(f"The prediction is: {output[0]}")


