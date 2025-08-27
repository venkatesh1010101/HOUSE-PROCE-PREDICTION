import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os


# Load model and scaler
with open('model.pkl', 'rb') as f:
    data = pickle.load(f)
model = data['model']
scaler = data['scaler']

# Preprocessing function for user input (make sure it matches training preprocessing)
def preprocess_input(input_df):
    # Map binary columns
    binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    for col in binary_cols:
        input_df[col] = input_df[col].map({'yes': 1, 'no': 0})

    # One hot encoding for furnishingstatus
    # Add columns for furnishingstatus as in training data
    for col in ['furnishingstatus_semi-furnished', 'furnishingstatus_unfurnished']:
        input_df[col] = 0

    if 'furnishingstatus' in input_df.columns:
        if input_df.loc[0, 'furnishingstatus'] == 'semi-furnished':
            input_df.loc[0, 'furnishingstatus_semi-furnished'] = 1
        elif input_df.loc[0, 'furnishingstatus'] == 'unfurnished':
            input_df.loc[0, 'furnishingstatus_unfurnished'] = 1

    # Drop original furnishingstatus column
    input_df = input_df.drop('furnishingstatus', axis=1)

    # Scale numeric features
    numeric_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking',
                    'furnishingstatus_semi-furnished', 'furnishingstatus_unfurnished']
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    return input_df

# Streamlit UI
st.title("🏠 House Price Prediction")

# Inputs from user
area = st.number_input('Area (in sq ft)', min_value=100, max_value=100000, value=7420)
bedrooms = st.number_input('Bedrooms', min_value=1, max_value=10, value=4)
bathrooms = st.number_input('Bathrooms', min_value=1, max_value=10, value=2)
stories = st.number_input('Stories', min_value=1, max_value=5, value=2)
mainroad = st.selectbox('Main Road Access', ['yes', 'no'])
guestroom = st.selectbox('Guest Room', ['yes', 'no'])
basement = st.selectbox('Basement', ['yes', 'no'])
hotwaterheating = st.selectbox('Hot Water Heating', ['yes', 'no'])
airconditioning = st.selectbox('Air Conditioning', ['yes', 'no'])
parking = st.number_input('Parking', min_value=0, max_value=10, value=2)
prefarea = st.selectbox('Preferred Area', ['yes', 'no'])
furnishingstatus = st.selectbox('Furnishing Status', ['furnished', 'semi-furnished', 'unfurnished'])

if st.button('Predict Price'):
    # Prepare input dataframe
    input_dict = {
        'area': [area],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'stories': [stories],
        'mainroad': [mainroad],
        'guestroom': [guestroom],
        'basement': [basement],
        'hotwaterheating': [hotwaterheating],
        'airconditioning': [airconditioning],
        'parking': [parking],
        'prefarea': [prefarea],
        'furnishingstatus': [furnishingstatus]
    }
    input_df = pd.DataFrame(input_dict)

    # Preprocess user input
    processed_input = preprocess_input(input_df)

    # Predict
    prediction = model.predict(processed_input)[0]

    st.success(f"Predicted House Price: ${prediction:,.2f}")
