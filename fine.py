import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from model import mobilevig  # pastikan path sesuai
from PIL import Image
import os

# ================== CONFIG ==================
NUM_CLASSES = 4
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-3
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CKPT_PATH = "m_MobileViG_s_50_epoch_0_d.pth"
SAVE_PATH = "mobilevig_finetuned_4cls.pth"

# ================== Dataset (contoh sederhana) ==================
class CoffeeDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

# Transformasi sesuai backbone
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ================== Load Model ==================
model = mobilevig.mobilevig_s(num_classes=1000)  # sama seperti checkpoint
checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]

# Filter backbone saja, jangan load head lama
model_state = model.state_dict()
filtered_state = {k: v for k, v in checkpoint.items() if k in model_state and v.shape == model_state[k].shape and "head" not in k}
model_state.update(filtered_state)
model.load_state_dict(model_state, strict=False)

# Ganti head
in_channels = getattr(model.head, "in_channels", 512)
model.head = nn.Conv2d(in_channels, NUM_CLASSES, kernel_size=1, bias=True)

# Freeze backbone
for name, param in model.named_parameters():
    if "head" not in name:
        param.requires_grad = False

model = model.to(DEVICE)

# ================== Optimizer & Loss ==================
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

import glob
import os
# ================== DataLoader ==================
# Contoh dummy data (ganti dengan dataset nyata)
dataset_dir = "..\\SKRIPSI FIX\\Dataset\\train\\"
# Ambil semua file jpg/png di subfolder kelas
image_paths = []
labels = []

for class_idx, class_name in enumerate(["Dark", "Green", "Light", "Medium"]):
    class_folder = os.path.join(dataset_dir, class_name)
    files = glob.glob(os.path.join(class_folder, "*.jpg")) + glob.glob(os.path.join(class_folder, "*.png"))
    image_paths.extend(files)
    labels.extend([class_idx] * len(files))
    
train_dataset = CoffeeDataset(image_paths, labels, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# ================== Fine-tuning Loop ==================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        if logits.dim() == 4:  # [B, C, 1, 1]
            logits = logits.flatten(2).squeeze(-1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

# ================== Simpan model hasil fine-tune ==================
torch.save(model.state_dict(), SAVE_PATH)
print(f"✅ Model fine-tuned disimpan di {SAVE_PATH}")
