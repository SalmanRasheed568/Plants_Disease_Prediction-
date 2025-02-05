import os
import streamlit as st
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import gdown

# Disable Streamlit file-watching to prevent conflicts
os.environ["STREAMLIT_WATCH_FILE"] = "false"

# Get the directory of the currently running script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load CSV files using relative paths
disease_info_path = os.path.join(BASE_DIR, 'disease_info.csv')
supplement_info_path = os.path.join(BASE_DIR, 'supplement_info.csv')

disease_info = pd.read_csv(disease_info_path, encoding='cp1252')
supplement_info = pd.read_csv(supplement_info_path, encoding='cp1252')

# Define file paths
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "plant_disease_model_Entire.pt")

# Google Drive file ID & Download URL
GDRIVE_FILE_ID = "11zdhHWZaN7cOs3HQFcvb_NMzWuMgMb9T"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

# Function to download the model if missing
def download_model():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    print("Downloading model file from Google Drive...")
    try:
        gdown.download(DOWNLOAD_URL, MODEL_PATH, fuzzy=True, quiet=False)
        print("Download complete.")
    except Exception as e:
        st.error(f"Model download failed: {e}")
        return False
    return True

# Ensure the model is available before proceeding
if not os.path.exists(MODEL_PATH):
    if not download_model():
        st.error("Failed to download the model. Please check your Google Drive permissions.")
        st.stop()

# Load the model with exception handling
try:
    model = CNN.CNN(39)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Function for image prediction
def prediction(image_path):
    try:
        image = Image.open(image_path)
        image = image.resize((224, 224))
        input_data = TF.to_tensor(image)
        input_data = input_data.view((-1, 3, 224, 224))
        output = model(input_data)
        output = output.detach().numpy()
        index = np.argmax(output)
        return index
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# Streamlit UI
st.title("🌱 Plant Disease Detection App")

# Embed HTML content from the files
def load_html(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        return f.read()

# Update the path to the HTML files
html_files_path = os.path.join(BASE_DIR, 'app', 'templates')

# List files in the directory for debugging
try:
    files = os.listdir(html_files_path)
    st.write("Files in templates directory:", files)
except Exception as e:
    st.error(f"Error listing files in directory: {e}")

# Load and display HTML files
html_files = [
    'contact us.html',
    'home.html',
    'index.html',
    'market.html',
    'submit.html'
]

for html_file in html_files:
    try:
        st.markdown(load_html(os.path.join(html_files_path, html_file)), unsafe_allow_html=True)
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")

uploaded_file = st.file_uploader("📤 Upload an image of the plant leaf", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)  # Ensure the temp directory exists

    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    pred = prediction(file_path)

    if pred is not None:
        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]

        st.subheader(f"🦠 Prediction: {title}")
        st.image(image_url, caption=title)
        st.write(f"**📄 Description:** {description}")
        st.write(f"**🛡 Prevention Steps:** {prevent}")

        st.subheader("💊 Recommended Supplement")
        st.image(supplement_image_url, caption=supplement_name)
        st.markdown(f"[🛒 Buy Here]({supplement_buy_link})")
    else:
        st.error("Failed to process the uploaded image.")