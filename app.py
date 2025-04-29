import streamlit as st
import tensorflow as tf
import pickle
from PIL import Image
import numpy as np

# Load model and tokenizer safely
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5", compile=False)

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
tokenizer = load_tokenizer()

# Utility: preprocess image
def preprocess_image(img):
    img = img.resize((299, 299))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

# Utility: generate caption (basic loop example)
def generate_caption(image):
    start_token = 'startseq'
    end_token = 'endseq'
    max_length = 34  # Adjust based on your model

    input_text = start_token
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([input_text])[0]
        sequence = tf.keras.preprocessing.sequence.pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image, sequence], verbose=0)
        word_index = np.argmax(yhat)
        word = tokenizer.index_word.get(word_index, None)
        if word is None or word == end_token:
            break
        input_text += ' ' + word
    return input_text.replace(start_token, '').strip()

# Streamlit UI
st.title("🖼️ Image Caption Generator")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Caption"):
        with st.spinner("Generating..."):
            preprocessed = preprocess_image(image)
            caption = generate_caption(preprocessed)
            st.success("Caption:")
            st.write(caption)
