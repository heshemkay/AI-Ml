import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Load your downloaded model
@st.cache_resource
def load_my_model():
    # Replace with 'best_model.h5'
    return tf.keras.models.load_model("best_model.h5")

model = load_my_model()

# 2. Build the Streamlit User Interface
st.title("🐱 vs 🐶 Cat or Dog Classifier")
st.write("Upload an image, and the AI model will tell you if it's a cat or a dog!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("🧠 Classifying...")

    # 3. Preprocess the image to match your model's training size
    # (Adjust 150, 150 to whatever size you used in Colab, e.g., 224, 224)
    img = image.resize((150, 150))
    img_array = np.array(img) / 255.0  # Normalize if you normalized in training
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # 4. Make prediction
    prediction = model.predict(img_array)
    
    # Adjust decoding logic based on your model's last layer output
    if prediction[0][0] > 0.5:
        st.success("🐶 It's a DOG!")
    else:
        st.success("🐱 It's a CAT!")
