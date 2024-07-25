import pickle
import streamlit as st

# Load model
def load_model():
    with open('random_forest_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Load encoders
def load_encoders():
    with open('label_encoders.pkl', 'rb') as file:
        encoders = pickle.load(file)
    return encoders

# encoders = load_encoders() # This line is causing the error because the file doesn't exist yet
# menu_category_encoder = encoders['MenuCategory']
# menu_item_encoder = encoders['MenuItem']
# profitability_encoder = encoders['Profitability']

# ----> Since the file doesn't exist, we'll create dummy encoders for now.
# ----> Remember to uncomment the `save_encoders()` function call at the end of the script and run it once to generate the 'label_encoders.pkl' file.
# ----> After running `save_encoders()`, comment it out again and uncomment the `encoders = load_encoders()` line.

from sklearn.preprocessing import LabelEncoder # Make sure LabelEncoder is imported

menu_category_encoder = LabelEncoder() 
menu_item_encoder = LabelEncoder()
profitability_encoder = LabelEncoder()

# ----> Example usage of the encoders to create some dummy classes (replace with your actual classes)
menu_category_encoder.fit(['Category A', 'Category B', 'Category C'])
menu_item_encoder.fit(['Item 1', 'Item 2', 'Item 3'])
profitability_encoder.fit(['High', 'Medium', 'Low']) 
# <---- 

# Title
st.title("Aplikasi Prediksi Profitabilitas Restoran")

# Input form
st.header("Input Data Restoran")
menu_category = st.selectbox("Kategori Menu", menu_category_encoder.classes_)
menu_item = st.selectbox("Nama Item Menu", menu_item_encoder.classes_)
ingredients = st.text_area("Bahan-bahan", "Contoh: ['Chicken', 'Fettuccine', 'Cheese']")
price = st.number_input("Harga", min_value=0.0, value=50.0)

# Preprocessing input
menu_category_encoded = menu_category_encoder.transform([menu_category])[0]
menu_item_encoded = menu_item_encoder.transform([menu_item])[0]

# Jika bahan-bahan digunakan sebagai input fitur, Anda harus mengubahnya menjadi vektor atau representasi numerik yang sesuai
# Dalam contoh ini, kita mengabaikan 'Ingredients' untuk prediksi
input_data = np.array([[menu_category_encoded, menu_item_encoded, price]])

# Prediction
if st.button("Prediksi"):
    prediction = model.predict(input_data)
    st.write(f"Hasil Prediksi: {profitability_encoder.inverse_transform(prediction)[0]}")

# Save encoders (run this code only once to save the encoders)
def save_encoders():
    encoders = {
        'MenuCategory': menu_category_encoder,
        'MenuItem': menu_item_encoder,
        'Profitability': profitability_encoder
    }
    with open('label_encoders.pkl', 'wb') as file:
        pickle.dump(encoders, file)

# Uncomment the line below and run once to save the encoders
save_encoders(