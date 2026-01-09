import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ========== KERAS (untuk verifikasi dengan InceptionV3) ==========
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ========== PYTORCH (untuk klasifikasi MobileViG) ==========
import torch
from torchvision.transforms import v2
from model import mobilevig  # pastikan path sesuai

# ================== CONFIG ==================
CLASSES = ["Dark", "Green", "Light", "Medium"]  # urutan sama dengan fine-tune
NUM_CLASSES = len(CLASSES)
IMG_SIZE = 224
FINETUNED_PATH = "mobilevig_finetuned_clean.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INCEPTION_H5_PATH = "model_Inception.h5"

# Inception config
EYE_LABEL_INDEX = 0
INCEPTION_THRESHOLD = 0.5

# Transformasi sama seperti training fine-tune
transform = v2.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ================== STREAMLIT CONFIG ==================
st.set_page_config(page_title="Klasifikasi Biji Kopi", page_icon="☕", layout="centered")
st.title("☕ Klasifikasi Tingkat Kematangan Sangrai Biji Kopi dengan GNN")

# ================== INCEPTION ==================
def preprocess_for_inception(image, target_size=(299, 299)):
    image = image.resize(target_size)
    arr = img_to_array(image)
    arr = tf.keras.applications.inception_v3.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

@st.cache_resource
def load_inception():
    try:
        model = load_model(INCEPTION_H5_PATH)
        return model, True
    except Exception as e:
        st.warning(f"⚠️ Model Inception tidak ditemukan/invalid: {e}. Verifikasi dilewati.")
        return None, False

inception_model, INCEPTION_OK = load_inception()

def is_valid_image(image, threshold=INCEPTION_THRESHOLD):
    if not INCEPTION_OK:
        return True
    batch = preprocess_for_inception(image)
    preds = inception_model.predict(batch, verbose=0)
    return preds[0][EYE_LABEL_INDEX] > threshold

# ================== LOAD MOBILEVIG_S FINE-TUNED ==================
@st.cache_resource
def load_mobilevig():
    # Buat model pretrained ImageNet 1000 kelas
    model = mobilevig.mobilevig_s(num_classes=1000)

    if not os.path.exists(FINETUNED_PATH):
        st.error(f"❌ Model fine-tuned tidak ditemukan: {FINETUNED_PATH}")
        st.stop()

    # Load checkpoint fine-tuned
    checkpoint = torch.load(FINETUNED_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    # Ganti head menjadi 4 kelas
    in_features = model.head.in_channels if hasattr(model.head, "in_channels") else 512
    model.head = torch.nn.Conv2d(in_features, NUM_CLASSES, kernel_size=1, bias=True)

    # Load checkpoint ke head dan layer lain jika cocok
    model_dict = model.state_dict()
    filtered_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)

    return model.to(DEVICE).eval()

torch_model = load_mobilevig()

# ================== PREDICT FUNCTION SAFE ==================
@torch.no_grad()
def predict_image(img: Image.Image):
    try:
        x = transform(img.convert("RGB")).unsqueeze(0).to(DEVICE)
        logits = torch_model(x)

        # Flatten output jika shape [B, C, 1, 1]
        if logits.dim() > 2:
            logits = logits.flatten(2).squeeze(-1)

        probs = torch.softmax(logits, dim=1).cpu().numpy()  # [1, NUM_CLASSES]

        # Validasi output
        if probs.shape[0] == 0 or probs.shape[1] != NUM_CLASSES:
            st.error(f"Output model tidak sesuai. Expected NUM_CLASSES={NUM_CLASSES}, got {probs.shape}")
            return None, None, None

        top_i = int(np.argmax(probs, axis=1)[0])
        return CLASSES[top_i], float(probs[0][top_i]), probs[0]

    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {e}")
        return None, None, None

# ================== STREAMLIT UI ==================

# Center alignment
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    st.image("green-11-_png.rf.bf3173dad6a550323a3f1b789ca7e756.jpg", 
             caption="Contoh upload gambar yang sesuai",
             width=200)

uploaded_file = st.file_uploader("Silahkan upload gambar biji kopi anda", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    if is_valid_image(image):
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.image(image, caption="✅ Gambar sesuai", width=350)

        if st.button("🔍 Proses"):
            label, conf, probs = predict_image(image)
            if label is not None:
                st.success(f"**Hasil:** {label} ({conf*100:.2f}% confidence)")

    else:
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