import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# Load the trained ResNet50 model for disease prediction
resnet_model = load_model("model_cnn_1.h5")

# Load the InceptionV3 model for eye image verification
inception_model = load_model("model_Inception.h5")

# Define classes and descriptions for ResNet50
classes = ["Dark", "Green", "Light", "Medium"]
descriptions = {
    "Dark": "Pembengkakan atau tonjolan mata yang disebabkan oleh gangguan pada otot mata atau jaringan.",
    "Green": "Kondisi keruh pada lensa mata yang menyebabkan penglihatan kabur, Segera konsultasikan ke dokter mata.",
    "Light": "Ketidaksejajaran mata yang bisa disebabkan oleh ketidakseimbangan otot, Sebaiknya konsultasikan ke dokter mata.",
    "Medium": "Mata dalam kondisi sehat tanpa indikasi penyakit, Tetap jaga kesehatan mata dengan rutin beristirahat."
}

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
eye_label_index = 0  
non_eye_label_index = 1  

# Function to check if image contains an eye using the InceptionV3 model
def is_eye_image(image, threshold=0.5):
    processed_image = preprocess_for_inception(image)
    predictions = inception_model.predict(processed_image)
    
    # Check if the model classifies the image as human eyes based on the threshold
    if predictions[0][eye_label_index] > threshold:
        return True  # Image is likely to be of human eyes
    return False  # Image is likely non-eye content

# Streamlit App
st.title("Prediksi Penyakit Mata dengan ResNet50")

# Show example image for uploading instructions
st.image("petunjuk_gambar.png", caption="Contoh upload gambar mata yang sesuai", use_container_width=True)

uploaded_image = st.file_uploader("Silahkan upload gambar mata anda 😊", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")

    # Check if the uploaded image is an eye image
    if is_eye_image(image):
        # Membuat kolom untuk memusatkan gambar
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col2:
            # Preview gambar di tengah dengan ukuran terkontrol
            st.image(image, caption="Gambar telah sesuai, Silahkan melanjutkan proses", width=350, use_container_width=False)
        
        # Preprocess the image and predict disease if it is an eye
        processed_image = preprocess_for_resnet(image)
        
        if st.button("Prediksi"):
            predictions = resnet_model.predict(processed_image)
            class_index = np.argmax(predictions)
            confidence = np.max(predictions) * 100
            predicted_class = classes[class_index]

            st.write(f"**Prediksi:** {predicted_class} ({confidence:.2f}% confidence)")
            #st.write("**Deskripsi Prediksi:**")
            #st.markdown(descriptions[predicted_class])
    else:
        # Membuat kolom untuk memusatkan gambar yang tidak valid
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col2:
            st.image(image, caption="Gambar yang anda upload tidak sesuai", width=350)
        
        st.error("❌❌ Gambar yang anda upload tidak sesuai, Silahkan upload gambar mata yang sesuai dengan contoh di atas ❌❌")

# Footer
def footer():
    st.markdown("<div class='footer'>© 2024 Muhammad Giat - 210013 - Eye👁️Check.AI </div>", unsafe_allow_html=True)

# Display footer
footer()