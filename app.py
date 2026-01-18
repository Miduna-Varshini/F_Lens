import os
# IMPORTANT: This must be at the very top to fix the 'batch_shape' error
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import numpy as np
from PIL import Image
import google.generativeai as genai
import gdown
import tensorflow as tf
from tensorflow.keras.models import load_model

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Fruit & Vegetable Recognition",
    page_icon="🍎",
    layout="centered"
)

# ---------------- CONFIG & PATHS ----------------
API_KEY = os.environ.get("GEMINI_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-pro")
else:
    st.error("Missing Gemini API Key. Please check Render Environment Variables.")

FRUIT_MODEL_PATH = "fruit_model.h5"
VEG_MODEL_PATH = "veg_model.h5"

# Google Drive File IDs
FRUIT_ID = "1WcgG4lM7G0-x6Q2h_JEV_sHUDACpW6TQ"
VEG_ID = "1OZvTjZZCv5PvRaAKdEikCWCk5_lWpRJ8"

# ---------------- HELPER FUNCTIONS ----------------

@st.cache_resource
def load_prediction_models():
    def download_if_missing(file_id, path):
        if not os.path.exists(path):
            with st.spinner(f"Downloading {path}..."):
                url = f'https://drive.google.com/uc?id={file_id}'
                gdown.download(url, path, quiet=False)
    
    download_if_missing(FRUIT_ID, FRUIT_MODEL_PATH)
    download_if_missing(VEG_ID, VEG_MODEL_PATH)
    
    # Using compile=False often avoids version-related loading issues
    f_model = load_model(FRUIT_MODEL_PATH, compile=False)
    v_model = load_model(VEG_MODEL_PATH, compile=False)
    return f_model, v_model

def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def get_nutrients(food, category):
    prompt = f"Act as a nutritionist. Provide the nutritional content of the {category} '{food}'. Include Calories, Vitamins, Minerals, and 3 Health benefits using bullet points. Keep it concise."
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except:
        return "⚠️ Nutrient data currently unavailable from Gemini AI."

# ---------------- APP LOGIC ----------------

st.title("🍎 Smart Food Recognition")
st.write("Upload an image to identify the fruit/vegetable and see its nutrients.")

try:
    fruit_model, veg_model = load_prediction_models()
    
    # Make sure these labels match your training data order!
    fruit_classes = ['Apple', 'Banana', 'Grapes', 'Mango', 'Orange', 'Pineapple', 'Strawberry', 'Watermelon']
    vegetable_classes = ['Beetroot', 'Cabbage', 'Carrot', 'Cauliflower', 'Potato', 'Tomato', 'Onion', 'Spinach']

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Target Image", use_container_width=True)

        with st.spinner("Analyzing image..."):
            img_array = preprocess_image(image)
            
            # Get predictions
            fruit_pred = fruit_model.predict(img_array, verbose=0)
            veg_pred = veg_model.predict(img_array, verbose=0)

            f_conf = np.max(fruit_pred)
            v_conf = np.max(veg_pred)

            # Logic to decide which category it belongs to
            if f_conf > v_conf:
                label = fruit_classes[np.argmax(fruit_pred)]
                category = "Fruit"
                confidence = f_conf
            else:
                label = vegetable_classes[np.argmax(veg_pred)]
                category = "Vegetable"
                confidence = v_conf

        # Display Result
        st.subheader(f"Identification: {label}")
        st.write(f"**Category:** {category}")
        st.write(f"**Confidence:** {confidence:.2%}")
        st.progress(float(confidence))

        # Show Nutrients
        st.divider()
        st.subheader(f"🥗 Nutritional Profile: {label}")
        with st.spinner("Fetching data..."):
            details = get_nutrients(label, category)
            st.markdown(details)

except Exception as e:
    st.error(f"System Error: {e}")

st.markdown("---")
st.caption("CNN Classification Models + Google Gemini Pro")
