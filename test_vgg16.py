import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet50

# Load the trained VGG16 model for disease prediction
#vgg16_model = load_model("cnn_gnn_hybrid_best.pth")

# Load the InceptionV3 model for eye image verification
inception_model = load_model("model_Inception.h5")

# Define classes and descriptions for VGG16
classes = ["Dark", "Green", "Light", "Medium"]
# descriptions = {
#     "Bulging Eyes": "Pembengkakan atau tonjolan mata yang disebabkan oleh gangguan pada otot mata atau jaringan.",
#     "Cataracts": "Kondisi keruh pada lensa mata yang menyebabkan penglihatan kabur, Segera konsultasikan ke dokter mata.",
#     "Crossed Eyes": "Ketidaksejajaran mata yang bisa disebabkan oleh ketidakseimbangan otot, Sebaiknya konsultasikan ke dokter mata.",
#     "Normal Eyes": "Mata dalam kondisi sehat tanpa indikasi penyakit, Tetap jaga kesehatan mata dengan rutin beristirahat.",
#     "Uveitis": "Peradangan pada lapisan tengah mata yang dapat menyebabkan kebutaan, Segera konsultasikan ke dokter mata."
# }

# Function to preprocess image for InceptionV3 model
def preprocess_for_inception(image, target_size=(299, 299)):
    image = image.resize(target_size)
    image_array = img_to_array(image)
    image_array = tf.keras.applications.inception_v3.preprocess_input(image_array)
    return np.expand_dims(image_array, axis=0)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_SIZE = 224

torch_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ---- GCN head minimal (tanpa library eksternal) ----
def build_grid_adjacency(H, W, radius=1, self_loop=True, device='cpu', dtype=torch.float32):
    N = H * W
    A = torch.zeros((N, N), dtype=dtype)
    def idx(r, c): return r * W + c
    for r in range(H):
        for c in range(W):
            u = idx(r, c)
            for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                for step in range(1, radius+1):
                    rr, cc = r + dr*step, c + dc*step
                    if 0 <= rr < H and 0 <= cc < W:
                        v = idx(rr, cc)
                        A[u, v] = 1.0
                        A[v, u] = 1.0
    if self_loop:
        A = A + torch.eye(N, dtype=dtype)
    deg = A.sum(dim=1)
    deg_inv_sqrt = torch.pow(torch.clamp(deg, min=1e-8), -0.5)
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt
    return A_hat.to(device=device)

class SimpleGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, dropout=0.0):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
    def forward(self, X, A_hat):
        # samakan dtype/device agar aman jika autocast aktif
        A_hat = A_hat.to(device=X.device, dtype=X.dtype)
        AX = torch.einsum('ij,bjk->bik', A_hat, X)  # (B,N,C_in)
        H = self.lin(AX)
        H = F.relu(H, inplace=True)
        H = self.dropout(H)
        return H

class FrozenResNetBackbone(nn.Module):
    def __init__(self, use_pretrained=False):
        super().__init__()
        # Pakai weights=None untuk mencegah download. Bobot akan terisi dari checkpoint.
        m = resnet50(weights=None if not use_pretrained else "DEFAULT")
        self.stem = nn.Sequential(
            m.conv1, m.bn1, m.relu, m.maxpool,  # 56x56
            m.layer1,                           # 56x56
            m.layer2,                           # 28x28
            m.layer3                            # 14x14
        )
        for p in self.stem.parameters():
            p.requires_grad = False
        self.out_channels = 1024
    def forward(self, x):
        return self.stem(x)  # (B,1024,H',W')

class CNN_GNN_Hybrid(nn.Module):
    def __init__(self, num_classes, gcn_hidden=256, gcn_dropout=0.1, radius=1):
        super().__init__()
        self.backbone = FrozenResNetBackbone(use_pretrained=False)
        self.gcn1 = SimpleGCNLayer(self.backbone.out_channels, gcn_hidden, dropout=gcn_dropout)
        self.gcn2 = SimpleGCNLayer(gcn_hidden, gcn_hidden, dropout=gcn_dropout)
        self.cls = nn.Linear(gcn_hidden, num_classes)
        self.radius = radius
        self.register_buffer("A_hat", None)
        self.grid_shape = None
    def _ensure_graph(self, feat):
        H, W = feat.shape[2], feat.shape[3]
        if self.grid_shape is None or self.grid_shape != (H, W) or self.A_hat is None:
            # simpan A_hat sebagai float32; nanti dicast di layer
            self.A_hat = build_grid_adjacency(H, W, radius=self.radius, device=feat.device, dtype=torch.float32)
            self.grid_shape = (H, W)
    def forward(self, x):
        feat = self.backbone(x)                          # (B,C,H',W')
        self._ensure_graph(feat)
        nodes = feat.flatten(2).transpose(1, 2).contiguous()  # (B,N,C)
        h = self.gcn1(nodes, self.A_hat)
        h = self.gcn2(h, self.A_hat)
        g = h.mean(dim=1)
        logits = self.cls(g)
        return logits

# Load checkpoint .pth (PYTORCH)
CLASSES_OVERRIDE = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load("cnn_gnn_hybrid_best.pth", map_location=device)
classes_from_ckpt = ckpt.get("classes", None)
CLASSES = CLASSES_OVERRIDE if CLASSES_OVERRIDE is not None else classes_from_ckpt
if CLASSES is None:
    st.error("Checkpoint .pth tidak menyimpan daftar kelas. Set CLASSES_OVERRIDE secara manual.")
    st.stop()

torch_model = CNN_GNN_Hybrid(num_classes=len(CLASSES), gcn_hidden=256, gcn_dropout=0.1, radius=1).to(device)
state = ckpt["state_dict"]
# buang 'A_hat' jika ada
state = {k: v for k, v in state.items() if k != "A_hat"}
torch_model.load_state_dict(state, strict=False)
torch_model.eval()

def preprocess_for_torch(image: Image.Image):
    return torch_tf(image.convert("RGB")).unsqueeze(0)

@torch.no_grad()
def predict_torch(image_tensor: torch.Tensor):
    image_tensor = image_tensor.to(device)
    logits = torch_model(image_tensor)
    probs = torch.softmax(logits, dim=1)[0]
    conf, idx = torch.max(probs, dim=0)
    return CLASSES[idx.item()], conf.item(), probs.cpu().numpy()

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
st.title("Prediksi Penyakit Mata dengan VGG16")

# Show example image for uploading instructions
st.image("petunjuk_gambar.png", caption="Contoh upload gambar mata yang sesuai", use_container_width=True)

uploaded_image = st.file_uploader("Silahkan upload gambar mata anda 😊", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")

    # Check if the uploaded image is an eye image
    if is_eye_image(image):
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.image(image, caption="Gambar telah sesuai, silakan lanjutkan", width=350, use_container_width=False)

        if st.button("Prediksi"):
            x = preprocess_for_torch(image)
            label, conf, _ = predict_torch(x)
            st.write(f"**Prediksi (PyTorch):** {label} ({conf*100:.2f}% confidence)")
    else:
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.image(image, caption="Gambar yang anda upload tidak sesuai", width=350)
        st.error("❌ Gambar tidak sesuai, silakan upload gambar mata sesuai contoh di atas ❌")

# Footer
def footer():
    st.markdown("<div class='footer'>© 2024 Muhammad Giat - 210013 - Eye👁️Check.AI </div>", unsafe_allow_html=True)

# Display footer
footer()