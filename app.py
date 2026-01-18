import os
# STEP 1: Force TensorFlow to use Legacy Keras (Keras 2)
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import numpy as np
from PIL import Image
import google.generativeai as genai
import gdown
import tensorflow as tf

# STEP 2: Use the specific tf_keras library for loading
# This bypasses the Keras 3 "batch_shape" error
import tf_keras as keras 

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Food Recognition", page_icon="🍎", layout="centered")

# ---------------- CONFIG & PATHS ----------------
API_KEY = os.environ.get("GEMINI_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-pro")
else:
    st.error("Missing Gemini API Key. Please check Render Environment Variables.")

FRUIT_MODEL_PATH = "fruit_model.h5"
VEG_MODEL_PATH = "veg_model.h5"
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
    
    # STEP 3: Load using tf_keras instead of standard tensorflow
    f_model = keras.models.load_model(FRUIT_MODEL_PATH, compile=False)
    v_model = keras.models.load_model(VEG_MODEL_PATH, compile=False)
    return f_model, v_model

def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def get_nutrients(food, category):
    prompt = f"Act as a nutritionist. List the nutritional content for 100g of {food} ({category}). Include Calories, Vitamins, and 3 Health benefits in bullets."
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except:
        return "⚠️ Nutrient data unavailable."

# ---------------- APP LOGIC ----------------

st.title("🍎 Smart Food Recognition")

try:
    fruit_model, veg_model = load_prediction_models()
    
    fruit_classes = ['Apple', 'Banana', 'Grapes', 'Mango', 'Orange', 'Pineapple', 'Strawberry', 'Watermelon']
    veg_classes = ['Beetroot', 'Cabbage', 'Carrot', 'Cauliflower', 'Potato', 'Tomato', 'Onion', 'Spinach']

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

        with st.spinner("Analyzing..."):
            img_array = preprocess_image(image)
            # Predict
            f_pred = fruit_model.predict(img_array, verbose=0)
            v_pred = veg_model.predict(img_array, verbose=0)

            # Compare confidence
            if np.max(f_pred) > np.max(v_pred):
                label, cat, conf = fruit_classes[np.argmax(f_pred)], "Fruit", np.max(f_pred)
            else:
                label, cat, conf = veg_classes[np.argmax(v_pred)], "Vegetable", np.max(v_pred)

        # Output logic
        st.success(f"### Identification: {label} ({cat})")
        st.write(f"Confidence: {conf:.2%}")
        
        st.divider()
        st.subheader(f"🥗 Nutrients: {label}")
        st.markdown(get_nutrients(label, cat))

except Exception as e:
    st.error(f"System Error: {e}")
