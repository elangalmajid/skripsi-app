import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

resnet_model = load_model("model_cnn_4.h5")

inception_model = load_model("model_Inception.h5")

# Define classes and descriptions for ResNet50
classes = ["Dark", "Green", "Light", "Medium"]

# Function to preprocess image for InceptionV3 model
def preprocess_for_inception(image, target_size=(299, 299)):
    image = image.resize(target_size)
    image_array = img_to_array(image)
    image_array = tf.keras.applications.inception_v3.preprocess_input(image_array)
    return np.expand_dims(image_array, axis=0)

# Function to preprocess image for ResNet50 model
def preprocess_for_resnet(image, target_size=(240, 240)):
    image = image.resize(target_size)
    image_array = img_to_array(image)
    image_array = tf.keras.applications.resnet50.preprocess_input(image_array)
    return np.expand_dims(image_array, axis=0)

# Define the labels as per the trained model
label_index = 0  
non_label_index = 1  

def is_image(image, threshold=0.5):
    processed_image = preprocess_for_inception(image)
    predictions = inception_model.predict(processed_image)
    
    if predictions[0][label_index] > threshold:
        return True  
    return False  

# Streamlit App
st.set_page_config(page_title="Klasifikasi Biji Kopi", page_icon="☕", layout="centered")
st.title("☕ Klasifikasi Tingkat Kematangan Sangrai Biji Kopi dengan CNN")

# Show example image for uploading instructions
# Center alignment
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    st.image("green-11-_png.rf.bf3173dad6a550323a3f1b789ca7e756.jpg", 
             caption="Contoh upload gambar yang sesuai",
             width=200)

uploaded_image = st.file_uploader("Silahkan upload gambar biji kopi anda", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")

    if is_image(image):
        # Membuat kolom untuk memusatkan gambar
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col2:
            # Preview gambar di tengah dengan ukuran terkontrol
            st.image(image, caption="✅ Gambar sesuai", width=350, use_container_width=False)
        
        processed_image = preprocess_for_resnet(image)
        
        if st.button("🔍 Proses"):
            predictions = resnet_model.predict(processed_image)
            class_index = np.argmax(predictions)
            confidence = np.max(predictions) * 100
            predicted_class = classes[class_index]

            st.success(f"**Hasil:** {predicted_class} ({confidence:.2f}% confidence)")
    else:
        # Membuat kolom untuk memusatkan gambar yang tidak valid
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col2:
            st.image(image, caption="❌ Gambar tidak sesuai", width=350)
        
        st.error("Gambar tidak sesuai, silakan upload gambar yang sesuai contoh di atas.")

# Footer
def footer():
    st.markdown("""
        <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            color: #F7F1F0;
            text-align: right;
            padding: 5px;
            font-size: small;
        }
        </style>
        <div class='footer'>© 2026 Elang Al Majid - 210002</div>
    """, unsafe_allow_html=True)

# Display footer
footer()