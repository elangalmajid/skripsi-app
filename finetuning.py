import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

# Import MobileViG
import sys
sys.path.append('model')  # Sesuaikan dengan path mobilevig.py
from mobilevig import mobilevig_ti, mobilevig_s

# ==================== CONFIG ====================
# Dataset
DATASET_DIR = "../skripsi/Dataset"  # Folder struktur: dataset/train/Dark, dataset/train/Green, etc.
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "valid")  # Atau test

# Classes
CLASS_NAMES = ["Dark", "Green", "Light", "Medium"]
NUM_CLASSES = len(CLASS_NAMES)

# Model
MODEL_TYPE = "s"  # "ti" atau "s"
PRETRAINED_PATH = None  # Path ke pretrained ImageNet weights (optional)
OUTPUT_PATH = "mobilevig_finetuned.pth"

# Training hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
IMG_SIZE = 224

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ==================== TRANSFORMS ====================
train_transforms = v2.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=15),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transforms = v2.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ==================== DATASET ====================
def load_datasets():
    print("\n📁 Loading datasets...")
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transforms)

    print(f"Classes: {train_dataset.classes}")
    print(f"Train images: {len(train_dataset)}")
    print(f"Val images: {len(val_dataset)}")

    # Verify classes match
    assert train_dataset.classes == CLASS_NAMES, f"Class mismatch! Expected {CLASS_NAMES}, got {train_dataset.classes}"

    # Use num_workers=0 for Windows compatibility
    num_workers = 0 if os.name == 'nt' else 4
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader

# ==================== MODEL ====================
def create_model():
    print("\n🔧 Creating model...")
    if MODEL_TYPE == "ti":
        model = mobilevig_ti(num_classes=1000)  # Start with ImageNet structure
        print("Model: MobileViG-Ti")
    elif MODEL_TYPE == "s":
        model = mobilevig_s(num_classes=1000)
        print("Model: MobileViG-S")
    else:
        raise ValueError("MODEL_TYPE must be 'ti' or 's'")

    # Load pretrained weights if available
    if PRETRAINED_PATH and os.path.exists(PRETRAINED_PATH):
        print(f"Loading pretrained weights from {PRETRAINED_PATH}")
        checkpoint = torch.load(PRETRAINED_PATH, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        model.load_state_dict(checkpoint, strict=False)

    # Modify head for 4 classes
    in_features = 512  # MobileViG embedding dimension
    model.head = nn.Conv2d(in_features, NUM_CLASSES, kernel_size=1, bias=True)
    if hasattr(model, 'dist_head'):
        model.dist_head = nn.Conv2d(in_features, NUM_CLASSES, kernel_size=1, bias=True)
        model.distillation = False  # Disable distillation for fine-tuning

    model = model.to(DEVICE)
    print(f"✅ Model ready with {NUM_CLASSES} classes")
    
    return model

# ==================== LOSS & OPTIMIZER ====================
def setup_training(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    return criterion, optimizer, scheduler

# ==================== TRAINING FUNCTIONS ====================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Handle distillation output if exists
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{running_loss/len(loader):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            
            # Handle distillation output if exists
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/len(loader):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# ==================== TRAINING LOOP ====================
def main():
    print("\n🚀 Starting training...")
    print(f"Epochs: {NUM_EPOCHS}, Batch size: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    print("=" * 70)
    
    # Load data
    train_loader, val_loader = load_datasets()
    
    # Create model
    model = create_model()
    
    # Setup training
    criterion, optimizer, scheduler = setup_training(model)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 70)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        # Update scheduler
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'class_names': CLASS_NAMES,
                'model_type': MODEL_TYPE
            }
            
            torch.save(checkpoint, OUTPUT_PATH)
            print(f"  ✅ Saved best model (Val Acc: {val_acc:.2f}%)")
        
        print("=" * 70)

    # ==================== TRAINING COMPLETE ====================
    print("\n" + "=" * 70)
    print("🎉 Training Complete!")
    print(f"Best Val Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"Model saved to: {OUTPUT_PATH}")
    print("=" * 70)

    # ==================== PLOT TRAINING HISTORY ====================
    print("\n📊 Generating training plots...")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Acc', marker='o')
    axes[1].plot(history['val_acc'], label='Val Acc', marker='s')
    axes[1].axhline(y=best_val_acc, color='r', linestyle='--', label=f'Best Val Acc: {best_val_acc:.2f}%')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("✅ Plots saved to: training_history.png")

    # ==================== LOAD & TEST BEST MODEL ====================
    print("\n🔍 Loading best model for final evaluation...")
    checkpoint = torch.load(OUTPUT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['state_dict'])

    final_val_loss, final_val_acc = validate(model, val_loader, criterion, DEVICE)
    print(f"\n📈 Final Validation Results:")
    print(f"  Loss: {final_val_loss:.4f}")
    print(f"  Accuracy: {final_val_acc:.2f}%")

    # ==================== SAVE STATE DICT ONLY (CLEANER) ====================
    print("\n💾 Saving clean state dict...")
    clean_path = OUTPUT_PATH.replace('.pth', '_clean.pth')
    torch.save({
        'state_dict': model.state_dict(),
        'class_names': CLASS_NAMES,
        'model_type': MODEL_TYPE,
        'val_acc': final_val_acc
    }, clean_path)
    print(f"✅ Clean model saved to: {clean_path}")

    print("\n✨ All done! You can now use this model for inference.")


if __name__ == '__main__':
    main()