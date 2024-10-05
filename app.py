import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the saved model
model = joblib.load('random_forest_model.pkl')

# Title for your app
st.title('Random Forest Classifier Deployment')

# Collect user input for each feature
st.header("Input the details for the loan prediction")

# Create widgets for each feature input
gender = st.selectbox("Gender", ['Male', 'Female'])
married = st.selectbox("Married", ['Yes', 'No'])
dependents = st.selectbox("Dependents", ['0', '1', '2', '3+'])
education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
self_employed = st.selectbox("Self Employed", ['Yes', 'No'])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_amount_term = st.number_input("Loan Amount Term", min_value=0)
credit_history = st.selectbox("Credit History", ['1', '0'])
property_area = st.selectbox("Property Area", ['Urban', 'Rural', 'Semiurban'])

# Button for prediction
if st.button("Predict Loan Status"):

    # Create a DataFrame for input data
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Married': [married],
        'Dependents': [dependents],
        'Education': [education],
        'Self_Employed': [self_employed],
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_amount_term],
        'Credit_History': [credit_history],
        'Property_Area': [property_area]
    })

    # Encode the categorical variables the same way as the training step
    categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']

    label_encoder = LabelEncoder()

    for col in categorical_cols:
        input_data[col] = label_encoder.fit_transform(input_data[col])

    # Predict loan status using the input data
    prediction = model.predict(input_data)

    # Display the prediction result
    if prediction == 1:
        st.success("Loan Approved!")
    else:
        st.error("Loan Denied.")

# Footer or additional details
st.write("Enter the details and click on 'Predict Loan Status' to get the prediction.")
