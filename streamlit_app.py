import streamlit as st
import pickle

st.title("predicting diabetes")

Pregnancies=st.number_input('feature_1',min_value=0,max_value=17)
Glucose	=st.number_input('feature_2',min_value=0,max_value=200)
BloodPressure=st.number_input('feature_3',min_value=0,max_value=122)
SkinThickness=st.number_input('feature_4',min_value=0,max_value=99)
Insulin=st.number_input('feature_5',min_value=0,max_value=846)
BMI=st.number_input('feature_6',min_value=0.0,max_value=67.1,value=1.0)
DiabetesPedigreeFunction=st.number_input('feature_7',min_value=0.0,max_value=2.42,value=1.0)
Age=st.number_input('feature_8',min_value=0,max_value=100)

with open('tree_cl.pkl', 'rb') as file:
    model=pickle.load(file)
output=model.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])


