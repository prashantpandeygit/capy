# app.py
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

st.set_page_config(page_title="Image Caption Generator", layout="centered")
st.title("🖼️ Image Caption Generator")

st.markdown(
    """
    Upload an image, and the app will generate a descriptive caption using a deep learning model.
    
    ⚠️ *Note: On first run, it might take up to a minute to download the model.*
    """
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Loading model and generating caption... please wait"):
        processor, model = load_model()
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

    st.markdown("### 📝 Generated Caption:")
    st.success(caption)
