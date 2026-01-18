import streamlit as st
import numpy as np
from PIL import Image
import google.generativeai as genai
import os
import urllib.request
from tensorflow.keras.models import load_model
# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Fruit & Vegetable Recognition",
    page_icon="🍎",
    layout="centered"
)

# ---------------- GEMINI CONFIG ----------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel("gemini-1.5-pro")

# ---------------- MODEL DOWNLOAD ----------------
FRUIT_MODEL_URL = "https://drive.google.com/uc?id=1WcgG4lM7G0-x6Q2h_JEV_sHUDACpW6TQ"
VEG_MODEL_URL   = "https://drive.google.com/uc?id=1OZvTjZZCv5PvRaAKdEikCWCk5_lWpRJ8"

FRUIT_MODEL_PATH = "fruit_model.h5"
VEG_MODEL_PATH = "veg_model.h5"

def download_model(url, path):
    if not os.path.exists(path):
        with st.spinner(f"Downloading {path}..."):
            urllib.request.urlretrieve(url, path)

download_model(FRUIT_MODEL_URL, FRUIT_MODEL_PATH)
download_model(VEG_MODEL_URL, VEG_MODEL_PATH)

# ---------------- LOAD MODELS ----------------
fruit_model = load_model(FRUIT_MODEL_PATH)
veg_model = load_model(VEG_MODEL_PATH)

# ---------------- CLASS LABELS ----------------
fruit_classes = [
    'Apple', 'Banana', 'Grapes', 'Mango',
    'Orange', 'Pineapple', 'Strawberry', 'Watermelon'
]

vegetable_classes = [
    'Beetroot', 'Cabbage', 'Carrot', 'Cauliflower',
    'Potato', 'Tomato', 'Onion', 'Spinach'
]

# ---------------- IMAGE PREPROCESS ----------------
def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ---------------- GEMINI NUTRIENT FUNCTION ----------------
def get_nutrients(food):
    prompt = f"""
    Give the nutritional content of {food}.

    Include:
    - Calories
    - Vitamins
    - Minerals
    - Health benefits

    Use bullet points.
    """
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except:
        return "⚠️ Nutrient data temporarily unavailable."

# ---------------- UI ----------------
st.title("🍎 Fruit & Vegetable Recognition System")
st.write("Upload an image to identify food and view nutrients")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = preprocess_image(image)

    fruit_pred = fruit_model.predict(img_array)
    veg_pred = veg_model.predict(img_array)

    fruit_conf = np.max(fruit_pred)
    veg_conf = np.max(veg_pred)

    if fruit_conf > veg_conf:
        idx = np.argmax(fruit_pred)
        label = fruit_classes[idx]
        category = "Fruit"
        confidence = fruit_conf
    else:
        idx = np.argmax(veg_pred)
        label = vegetable_classes[idx]
        category = "Vegetable"
        confidence = veg_conf

    st.success(f"### 🧠 Detected: {label} ({category})")
    st.write(f"**Confidence:** {confidence:.2f}")
    st.progress(float(confidence))

    with st.spinner("🤖 Fetching nutrient details..."):
        nutrients = get_nutrients(label)

    st.subheader("🥗 Nutritional Information")
    st.write(nutrients)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("CNN Models + Gemini AI | College Mini Project")
