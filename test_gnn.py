# test_gnn.py — versi Streamlit dengan verifikasi Inception seperti test_vgg16.py

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

# ========== ARSITEKTUR ==========
# Pastikan model/mobilevig.py ada, dan timm/einops terpasang:
# pip install timm==0.9.16 einops
from model import mobilevig

# ================== CONFIG ==================
# Kelas target (urutannya harus sama dgn training)
CLASSES = ["Dark", "Green", "Light", "Medium"]
NUM_CLASSES = len(CLASSES)

# Checkpoint MobileViG yang sudah dibersihkan (CPU-only)
CKPT_PATH = "mobilevig_state_cpu_only.pth"

# InceptionV3 (Keras) untuk verifikasi gambar (mis. gambar valid)
INCEPTION_H5_PATH = "model_Inception.h5"
EYE_LABEL_INDEX = 0      # sesuaikan: index kelas "valid" pada model Inception kamu
INCEPTION_THRESHOLD = 0.5

# Transform PyTorch (samakan dgn training/test)
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
torch_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

st.set_page_config(page_title="Prediksi (Inception + MobileViG)", page_icon="🧠", layout="centered")
st.title("Prediksi dengan Inception (verifikasi) + MobileViG (klasifikasi)")

# ================== INCEPTION (KERAS) ==================
def preprocess_for_inception(image, target_size=(299, 299)):
    image = image.resize(target_size)
    arr = img_to_array(image)
    arr = tf.keras.applications.inception_v3.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def load_inception():
    try:
        model = load_model(INCEPTION_H5_PATH)
        return model, True
    except Exception as e:
        st.warning(f"Model Inception tidak ditemukan/invalid: {e}. Verifikasi akan dilewati.")
        return None, False

inception_model, INCEPTION_OK = load_inception()

def is_valid_image(image, threshold=INCEPTION_THRESHOLD):
    """Jika Inception tersedia, cek probabilitas kelas 'valid' (index EYE_LABEL_INDEX)."""
    if not INCEPTION_OK:
        return True
    batch = preprocess_for_inception(image)
    preds = inception_model.predict(batch, verbose=0)
    return preds[0][EYE_LABEL_INDEX] > threshold

# ================== DEVICE PICKER ==================
def pick_device():
    try:
        import torch_directml as tdm
        return tdm.device()
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = pick_device()

# ================== PAKSA HEAD = NUM_CLASSES ==================
def force_num_classes(model: nn.Module, num_classes: int) -> nn.Module:
    # timm style
    if hasattr(model, "reset_classifier"):
        try:
            model.reset_classifier(num_classes)
            return model
        except Exception:
            pass
    # .classifier
    if hasattr(model, "classifier") and isinstance(getattr(model, "classifier"), nn.Linear):
        in_f = model.classifier.in_features
        model.classifier = nn.Linear(in_f, num_classes);  return model
    # .head.fc
    if hasattr(model, "head") and hasattr(model.head, "fc") and isinstance(model.head.fc, nn.Linear):
        in_f = model.head.fc.in_features
        model.head.fc = nn.Linear(in_f, num_classes);     return model
    # .head
    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        in_f = model.head.in_features
        model.head = nn.Linear(in_f, num_classes);        return model
    # .fc
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes);          return model
    return model

class HeadAdapter(nn.Module):
    """Jika base model keluarkan C!=NUM_CLASSES, adaptasikan ke NUM_CLASSES."""
    def __init__(self, base: nn.Module, in_classes: int, out_classes: int):
        super().__init__()
        self.base = base
        self.adapter = nn.Linear(in_classes, out_classes)
    def forward(self, x):
        return self.adapter(self.base(x))

# ================== LOAD MOBILEVIG (CACHE) ==================
@st.cache_resource(show_spinner=True)
def load_mobilevig():
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {os.path.abspath(CKPT_PATH)}")
    model = mobilevig.mobilevig_ti(num_classes=NUM_CLASSES)
    model = force_num_classes(model, NUM_CLASSES)
    state = torch.load(CKPT_PATH, map_location="cpu")
    msd = model.state_dict()
    # hanya load kunci yang cocok (head 1000 akan otomatis di-skip)
    filtered = {k: v for k, v in state.items() if (k in msd and msd[k].shape == v.shape)}
    msd.update(filtered)
    model.load_state_dict(msd, strict=False)
    model.to(device).eval()

    # cek dimensi keluaran
    with torch.no_grad():
        dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=device)
        out = model(dummy)
    out_dim = out.shape[-1]
    if out_dim != NUM_CLASSES:
        model = HeadAdapter(model, in_classes=out_dim, out_classes=NUM_CLASSES).to(device).eval()
    return model

try:
    torch_model = load_mobilevig()
except Exception as e:
    st.error(f"Gagal memuat model MobileViG: {e}")
    st.stop()

# ================== INFERENCE ==================
@torch.no_grad()
def predict_torch(img_pil: Image.Image):
    x = torch_tf(img_pil.convert("RGB")).unsqueeze(0).to(device)
    logits = torch_model(x)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    top_i = int(np.argmax(probs))
    names = CLASSES[:len(probs)] if len(CLASSES) >= len(probs) else [f"Class {k}" for k in range(len(probs))]
    top_name = names[top_i] if top_i < len(names) else f"Class {top_i}"
    return top_name, float(probs[top_i]), probs, names

# ================== UI (meniru pola test_vgg16.py) ==================
st.image("petunjuk_gambar.png", caption="Contoh upload gambar yang sesuai", use_container_width=True)
uploaded_image = st.file_uploader("Silahkan upload gambar mata anda 😊", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")

    # Langkah verifikasi dengan Inception (ala test_vgg16)
    if is_valid_image(image):
        # pusatkan preview di kolom tengah
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.image(image, caption="Gambar telah sesuai, silakan lanjutkan", width=350, use_container_width=False)

        if st.button("Prediksi"):
            try:
                label, conf, probs, names = predict_torch(image)

                # Info mismatch (kalau ada)
                if len(names) != len(CLASSES):
                    st.warning(f"Jumlah label ({len(CLASSES)}) ≠ output model ({len(names)}). Menampilkan sesuai output model.")

                st.success(f"Prediksi (MobileViG): **{label}** ({conf*100:.2f}% confidence)")

                st.subheader("Probabilitas per kelas")
                df = pd.DataFrame({"Kelas": names, "Probabilitas (%)": (probs * 100).round(2)})
                st.table(df.head(20))
                if len(df) > 20:
                    st.caption(f"Menampilkan 20 dari {len(df)} kelas.")

                # Bar chart top-10
                topk = min(10, len(names))
                idxs = np.argsort(-probs)[:topk]
                st.bar_chart(pd.DataFrame({"Prob": probs[idxs]}, index=[names[i] for i in idxs]))
            except Exception as e:
                st.error(f"Terjadi kesalahan saat inferensi: {e}")
    else:
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.image(image, caption="Gambar yang anda upload tidak sesuai", width=350)
        st.error("❌ Gambar tidak sesuai, silakan upload gambar yang sesuai contoh di atas ❌")

# ================== FOOTER ==================
def footer():
    st.markdown("<div class='footer'>© 2025 — Inception + MobileViG Inference</div>", unsafe_allow_html=True)
footer()
