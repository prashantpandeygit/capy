import streamlit as st
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch

@st.cache_resource
def load_model():
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    return model, processor, tokenizer

model, processor, tokenizer = load_model()

def generate_caption(image):
    if image.mode != "RGB":
        image = image.convert("RGB")

    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    output_ids = model.generate(
        pixel_values,
        max_length=20,
        num_beams=3,
        do_sample=False,
        early_stopping=True
    )

    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    caption = caption.replace("_", " ").strip().capitalize()
    return caption

st.set_page_config(page_title="Image Caption Generator", layout="centered")

uploaded_image = st.file_uploader("Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Generating a caption..."):
        caption = generate_caption(image)
        st.success("Here’s your caption:")
        st.markdown(f"### \"{caption}\"")

        st.download_button(
            label="📄 Download Caption as .txt",
            data=caption,
            file_name="caption.txt",
            mime="text/plain"
        )
