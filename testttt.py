# from tensorflow.keras.models import load_model

# # load model
# model = load_model("model_cnn_4.h5")

# # tampilkan arsitektur
# model.summary()

# # simpan arsitektur ke file JSON
# with open("model_architecture.json", "w") as f:
#     f.write(model.to_json())

import torch
from model import mobilevig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Buat arsitektur dasar (1000 kelas default)
model = mobilevig.mobilevig_s(dropout=0.3).to(device)

# Override head agar sesuai 4 kelas
model.head = torch.nn.Conv2d(512, 4, kernel_size=1, stride=1)

# Load state_dict
state_dict = torch.load("mobilevig_finetuned_4cls.pth", map_location=device)
model.load_state_dict(state_dict)

model.eval()

print("Berhasil load model!")
