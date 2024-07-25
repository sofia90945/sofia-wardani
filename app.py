import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Load model
def load_model():
    with open('random_forest_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Title
st.title("Aplikasi Prediksi Profitabilitas Restoran")

# Input form
st.header("Input Data")
menu_category = st.selectbox("Menu Category", ["Beverages", "Appetizers", "Desserts", "Main Course"])
menu_item = st.text_input("Menu Item")
ingredients = st.text_input("Ingredients")
price = st.number_input("Price", min_value=0.0, value=10.0)

# Preprocessing input
label_encoder = LabelEncoder()
encoded_menu_category = label_encoder.fit_transform([menu_category])[0]
encoded_menu_item = label_encoder.fit_transform([menu_item])[0]
encoded_ingredients = label_encoder.fit_transform([ingredients])[0]

input_data = np.array([[encoded_menu_category, encoded_menu_item, encoded_ingredients, price]])

# Prediction
if st.button("Prediksi"):
    prediction = model.predict(input_data)
    st.write(f"Hasil Prediksi: {'High' if prediction[0] == 1 else 'Low'}")
