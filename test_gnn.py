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
import torch.nn as nn
from torchvision import transforms
from model import mobilevig  # pastikan path sesuai

# ================== CONFIG ==================
CLASSES = ["Dark", "Green", "Light", "Medium"]   # urutan harus sama dengan training
NUM_CLASSES = len(CLASSES)
IMG_SIZE = 224

# Path model
CKPT_PATH = "mobilevig_cpu_clean.pth"
INCEPTION_H5_PATH = "model_Inception.h5"

# Inception config
EYE_LABEL_INDEX = 0      # index kelas valid pada Inception
INCEPTION_THRESHOLD = 0.5

# Transform (samakan dengan training)
torch_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),  # jika training pakai rescale=1./255
    # Jika training pakai normalisasi ImageNet, aktifkan ini:
    # transforms.Normalize([0.485, 0.456, 0.406],
    #                      [0.229, 0.224, 0.225]),
])

# ================== STREAMLIT CONFIG ==================
st.set_page_config(page_title="Klasifikasi Biji Kopi", page_icon="☕", layout="centered")
st.title("☕ Klasifikasi Tingkat Kematangan Sangrai Biji Kopi dengan GNN")
# st.write("Upload gambar → diverifikasi dengan **InceptionV3** → diprediksi dengan **MobileViG**")

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

# ================== LOAD MOBILEVIG ==================
@st.cache_resource
def load_mobilevig():
    if not os.path.exists(CKPT_PATH):
        st.error(f"❌ Checkpoint tidak ditemukan: {CKPT_PATH}")
        st.stop()

    model = mobilevig.mobilevig_s(num_classes=NUM_CLASSES)
    state = torch.load(CKPT_PATH, map_location="cpu")

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    msd = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in msd and v.shape == msd[k].shape}
    msd.update(filtered)
    model.load_state_dict(msd, strict=False)
    model.eval()
    return model

torch_model = load_mobilevig()

# ================== PREDICT ==================
@torch.no_grad()
def predict_image(img: Image.Image):
    x = torch_tf(img.convert("RGB")).unsqueeze(0)
    logits = torch_model(x)
    probs = torch.softmax(logits, dim=1)[0].numpy()

    top_i = int(np.argmax(probs))
    return CLASSES[top_i], float(probs[top_i]), probs

# ================== STREAMLIT UI ==================
uploaded_file = st.file_uploader("📤 Upload gambar biji kopi", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    # st.image(image, caption="Gambar diupload", use_container_width=True)
    
    # 1️⃣ Verifikasi otomatis dengan Inception
    if is_valid_image(image):
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.image(image, caption="✅ Gambar sesuai", width=350)

        if st.button("🔍 Proses"):
            try:
                label, conf, probs = predict_image(image)

                st.success(f"**Hasil:** {label} ({conf*100:.2f}%)")

                    # df = pd.DataFrame({
                    #     "Kelas": CLASSES,
                    #     "Confidence (%)": (probs * 100).round(2)
                    # })
                    

            except Exception:
                pass
    else:
        # kalau tidak valid oleh Inception
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.image(image, caption="❌ Gambar tidak sesuai", width=350)
        st.error("Gambar tidak sesuai, silakan upload gambar yang sesuai contoh di atas.")
            




# ================== FOOTER ==================
# def footer():
#     st.markdown("<div class='footer'>© 2025 — Inception + MobileViG Inference</div>", unsafe_allow_html=True)
# footer()
