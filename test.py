# test_model_debug.py
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from model import mobilevig  # pastikan path sesuai

# ================= CONFIG =================
CKPT_PATH = "mobilevig_cpu_clean.pth"
CLASSES = ["Dark", "Green", "Light", "Medium"]   # sesuaikan urutan dengan training
NUM_CLASSES = len(CLASSES)
IMG_SIZE = 224

# Transform (samakan dengan training!)
torch_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    # Kalau training cuma pakai rescale=1./255, ganti ini jadi None
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ================= LOAD MODEL =================
def load_model():
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

# ================= PREDICT =================
def predict_image(path):
    img = Image.open(path).convert("RGB")
    x = torch_tf(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].numpy()

    print("\n=== HASIL PREDIKSI ===")
    for i, cls in enumerate(CLASSES):
        print(f"{cls:>7}: {probs[i]*100:.2f}%")
    top_i = int(np.argmax(probs))
    print(f"\n>> Prediksi utama: {CLASSES[top_i]} ({probs[top_i]*100:.2f}%)")

# ================= MAIN =================
if __name__ == "__main__":
    model = load_model()
    test_image = "light-127-_png.rf.d15ec1c8f725b93b2444519bd12167ad.jpg"  # ganti dengan path gambar nyata
    predict_image(test_image)
