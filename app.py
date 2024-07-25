import streamlit as st
import numpy as np
import pickle

# Load model and label encoders
def load_model():
    with open('random_forest_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def load_label_encoders():
    with open('label_encoders.pkl', 'rb') as file:
        label_encoders = pickle.load(file)
    return label_encoders

model = load_model()
label_encoders = load_label_encoders()

# Title
st.title("Aplikasi Prediksi Profitabilitas Restoran")

# Input form
st.header("Input Data")

# Display dropdowns for Menu Category, Menu Item, and Ingredients
menu_category = st.selectbox("Menu Category", label_encoders['menu_category'].classes_)
menu_item = st.selectbox("Menu Item", label_encoders['menu_item'].classes_)
ingredients = st.selectbox("Ingredients", label_encoders['ingredients'].classes_)
price = st.number_input("Price", min_value=0.0, value=10.0)

# Preprocessing input
def encode_input(value, encoder, encoder_name):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        st.warning(f"{encoder_name} '{value}' tidak dikenal. Gunakan nilai yang valid.")
        return None

encoded_menu_category = encode_input(menu_category, label_encoders['menu_category'], 'Menu Category')
encoded_menu_item = encode_input(menu_item, label_encoders['menu_item'], 'Menu Item')
encoded_ingredients = encode_input(ingredients, label_encoders['ingredients'], 'Ingredients')

if None not in [encoded_menu_category, encoded_menu_item, encoded_ingredients]:
    input_data = np.array([[encoded_menu_category, encoded_menu_item, encoded_ingredients, price]])

    # Prediction
    if st.button("Prediksi"):
        try:
            prediction = model.predict(input_data)
            st.write(f"Hasil Prediksi: {'High' if prediction[0] == 1 else 'Low'}")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
else:
    st.write("Mohon periksa input Anda dan coba lagi.")
