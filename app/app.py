import os
import streamlit as st
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms.functional as TF
import CNN

# Load disease and supplement information
disease_info = pd.read_csv(r'C:\Users\Dubai Computers\PycharmProjects\PythonProject2\app\disease_info.csv',
                           encoding='cp1252')
supplement_info = pd.read_csv(r'C:\Users\Dubai Computers\PycharmProjects\PythonProject2\app\supplement_info.csv',
                              encoding='cp1252')

# Load the model
model = CNN.CNN(39)
model.load_state_dict(
    torch.load(r'C:\Users\Dubai Computers\New folder\model\plant_disease_model_Entire.pt'))
model.eval()


def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index


# Streamlit UI
st.title("Plant Disease Detection App")

uploaded_file = st.file_uploader("Upload an image of the plant leaf", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)  # Ensure the temp directory exists

    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    pred = prediction(file_path)
    title = disease_info['disease_name'][pred]
    description = disease_info['description'][pred]
    prevent = disease_info['Possible Steps'][pred]
    image_url = disease_info['image_url'][pred]
    supplement_name = supplement_info['supplement name'][pred]
    supplement_image_url = supplement_info['supplement image'][pred]
    supplement_buy_link = supplement_info['buy link'][pred]

    st.subheader(f"Prediction: {title}")
    st.image(image_url, caption=title)
    st.write(f"**Description:** {description}")
    st.write(f"**Prevention Steps:** {prevent}")

    st.subheader("Recommended Supplement")
    st.image(supplement_image_url, caption=supplement_name)
    st.markdown(f"[Buy Here]({supplement_buy_link})")
