import streamlit as st
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch

# Load model and tokenizer
@st.cache_resource
def load_model():
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    return model, processor, tokenizer

model, processor, tokenizer = load_model()

# Caption generation function (no beam search)
def generate_caption(image):
    if image.mode != "RGB":
        image = image.convert(mode="RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    output_ids = model.generate(pixel_values, max_length=16)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption



uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Generating caption..."):
        caption = generate_caption(image)
        st.success("Caption Generated!")
        st.markdown(f"**Caption:** _{caption}_")

        # Option to download caption
        st.download_button(
            label="📄 Download Caption as .txt",
            data=caption,
            file_name="caption.txt",
            mime="text/plain"
        )
