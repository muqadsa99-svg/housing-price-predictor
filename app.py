import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load the trained model
model = joblib.load('house_price_model.pkl')

# 2. App Title and Description
st.title("🏡 Advanced House Price Prediction")
st.write("""
This app uses a **Stacking Regressor** (Ridge, Lasso, XGBoost, LightGBM) to predict house prices.
Adjust the parameters below to see the estimated value.
""")

# 3. Sidebar Inputs
st.sidebar.header("House Details")

def user_input_features():
    # We need to initialize a DataFrame with the same columns as the training data
    # Load the sample to get columns/structure
    sample_df = pd.read_csv('sample_input.csv')
    input_df = sample_df.copy()

    # Reset all values to 0 or mode for safety
    for col in input_df.columns:
        if input_df[col].dtype == 'object':
            input_df[col] = input_df[col].mode()[0]
        else:
            input_df[col] = 0

    # Create widgets for Key Features
    # Note: Ideally, you would create widgets for all features.
    # Here we pick the most important ones for the demo.

    LotArea = st.sidebar.number_input("Lot Area (sq ft)", min_value=1000, max_value=50000, value=10000)
    OverallQual = st.sidebar.slider("Overall Quality (1-10)", 1, 10, 5)
    YearBuilt = st.sidebar.number_input("Year Built", 1900, 2023, 2000)
    TotalBsmtSF = st.sidebar.number_input("Total Basement (sq ft)", 0, 3000, 1000)
    GrLivArea = st.sidebar.number_input("Above Ground Living Area (sq ft)", 500, 5000, 1500)
    GarageCars = st.sidebar.slider("Garage Cars", 0, 4, 2)

    # Update the input dataframe
    input_df['LotArea'] = LotArea
    input_df['OverallQual'] = OverallQual
    input_df['YearBuilt'] = YearBuilt
    input_df['TotalBsmtSF'] = TotalBsmtSF
    input_df['GrLivArea'] = GrLivArea
    input_df['GarageCars'] = GarageCars
    input_df['1stFlrSF'] = GrLivArea * 0.6  # Rough approximation for demo if not asking explicitly
    input_df['2ndFlrSF'] = GrLivArea * 0.4

    return input_df

# 4. Get User Input
input_data = user_input_features()

# 5. Show User Input
st.subheader("Selected Parameters")
st.write(input_data[['LotArea', 'OverallQual', 'YearBuilt', 'GrLivArea']])

# 6. Prediction Logic
if st.button("Predict Price"):
    try:
        # The pipeline handles raw data, scaling, and encoding automatically!
        # Prediction is in Log scale, so we reverse it using np.expm1
        log_prediction = model.predict(input_data)
        price = np.expm1(log_prediction)

        st.success(f"Estimated Sale Price: ${price[0]:,.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")