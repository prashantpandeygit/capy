import streamlit as st
import tensorflow as tf
import pickle
from PIL import Image
import numpy as np

# Load InceptionV3 feature extractor
@st.cache_resource
def load_feature_extractor():
    base_model = tf.keras.applications.InceptionV3(include_top=False, pooling='avg')
    return tf.keras.Model(inputs=base_model.input, outputs=base_model.output)

# Load model and tokenizer
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5", compile=False)

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
tokenizer = load_tokenizer()
feature_extractor = load_feature_extractor()

# Preprocess uploaded image and extract features
def preprocess_and_extract(img):
    img = img.resize((299, 299))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return feature_extractor.predict(img_array)

# Generate caption
def generate_caption(image_features):
    start_token = 'startseq'
    end_token = 'endseq'
    max_length = 34

    input_text = start_token
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([input_text])[0]
        sequence = tf.keras.preprocessing.sequence.pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image_features, sequence], verbose=0)
        word_index = np.argmax(yhat)
        word = tokenizer.index_word.get(word_index)
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
            features = preprocess_and_extract(image)
            caption = generate_caption(features)
            st.success("Caption:")
            st.write(caption)
