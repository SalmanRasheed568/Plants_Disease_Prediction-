import os
import streamlit as st
import torch
import numpy as np
import pandas as pd
import base64
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import gdown
import random

# Disable Streamlit file-watching
os.environ["STREAMLIT_WATCH_FILE"] = "false"

# Get the directory of the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load CSV files
disease_info = pd.read_csv(os.path.join(BASE_DIR, 'disease_info.csv'), encoding='cp1252')
supplement_info = pd.read_csv(os.path.join(BASE_DIR, 'supplement_info.csv'), encoding='cp1252')

# Define file paths
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "plant_disease_model_Entire.pt")

# Google Drive model download
GDRIVE_FILE_ID = "11zdhHWZaN7cOs3HQFcvb_NMzWuMgMb9T"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

def download_model():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    gdown.download(DOWNLOAD_URL, MODEL_PATH, fuzzy=True, quiet=False)

if not os.path.exists(MODEL_PATH):
    download_model()

# Load the model
model = CNN.CNN(39)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

def predict(image_path):
    image = Image.open(image_path).resize((224, 224))
    input_data = TF.to_tensor(image).view((-1, 3, 224, 224))
    output = model(input_data).detach().numpy()
    confidence = np.max(output) * 100
    return np.argmax(output), confidence, output

def set_background(image_path):
    with open(image_path, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
            color: white !important;
        }}
        </style>
    """, unsafe_allow_html=True)

background_image_path = "app/102927372-plant-4k-wallpaper.jpg"

# Check if file exists
if os.path.isfile(background_image_path):
    set_background(background_image_path)
else:
    st.error("⚠️ Background image not found! Please check the file path or move the image to the correct location.")

st.title("🌱 Plant Disease Detection App")

uploaded_files = st.file_uploader("📤 Upload leaf images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    for image_file in uploaded_files:
        st.image(image_file, caption="Uploaded Image", use_container_width=True)
        file_path = os.path.join("temp", image_file.name)
        with open(file_path, "wb") as f:
            f.write(image_file.getbuffer())

        with st.spinner("🔍 Analyzing..."):
            pred, confidence, all_probs = predict(file_path)

        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        image_url = disease_info['image_url'][pred]

        st.subheader(f"🦠 Prediction: {title}")
        st.image(image_url, caption=title)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            title={'text': "Confidence Score"},
            gauge={'axis': {'range': [0, 100]}}
        ))
        st.plotly_chart(fig)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("📄 Description")
            st.write(description)
        with col2:
            st.subheader("🛡 Prevention Steps")
            st.write(prevent)
        with col3:
            st.subheader("💊 Recommended Supplement")
            st.markdown(f"[🛒 Buy Here]({supplement_buy_link})", unsafe_allow_html=True)

        st.subheader("🔍 Alternative Predictions")
        alt_preds = np.argsort(all_probs[0])[-3:][::-1]
        for alt_pred in alt_preds:
            st.write(f"- {disease_info['disease_name'][alt_pred]} ({all_probs[0][alt_pred]*100:.2f}% confidence)")

        fig = px.bar(x=disease_info['disease_name'][alt_preds], y=all_probs[0][alt_preds] * 100,
                     labels={'x': 'Disease', 'y': 'Confidence (%)'}, title="Prediction Confidence Levels")
        st.plotly_chart(fig)

# Amazing Feature: Random Plant Health Tips
plant_tips = [
    "💧 Water your plants in the morning to prevent fungal growth.",
    "🌞 Ensure plants get enough sunlight but avoid direct afternoon heat.",
    "🍂 Remove dead leaves to promote healthy growth.",
    "🌱 Rotate potted plants weekly for balanced growth.",
    "🐞 Introduce beneficial insects like ladybugs to control pests naturally.",
    "🌿 Use organic fertilizers like compost for better soil health.",
    "🛡 Apply neem oil to protect plants from fungal diseases.",
    "🔄 Change soil every season to avoid nutrient depletion.",
    "🥦 Grow companion plants together for mutual benefits, like basil and tomatoes!"
]

st.sidebar.title("🌟 Plant Care Tip of the Day")
st.sidebar.info(random.choice(plant_tips))

st.sidebar.title("🌿 About the App")
st.sidebar.info("This app detects plant diseases from images and provides prevention steps and supplement recommendations.")
