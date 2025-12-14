import torch
import os

# ==================== CONFIG ====================
# Input checkpoint path
CHECKPOINT_PATH = "mobilevig_finetuned.pth"

# Output clean path
CLEAN_PATH = CHECKPOINT_PATH.replace('.pth', '_clean.pth')

# Model info (sesuaikan dengan training Anda)
CLASS_NAMES = ["Dark", "Green", "Light", "Medium"]
MODEL_TYPE = "ti"  # "ti" atau "s"

# ==================== LOAD CHECKPOINT ====================
print("=" * 70)
print("🔄 Converting Checkpoint to Clean Version")
print("=" * 70)

if not os.path.exists(CHECKPOINT_PATH):
    print(f"❌ Error: Checkpoint not found at {CHECKPOINT_PATH}")
    print("Please make sure the file exists!")
    exit(1)

print(f"\n📂 Loading checkpoint from: {CHECKPOINT_PATH}")
checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')

# ==================== EXTRACT INFO ====================
if isinstance(checkpoint, dict):
    print("\n✅ Checkpoint is a dictionary")
    print(f"Keys in checkpoint: {list(checkpoint.keys())}")
    
    # Extract state_dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("✅ Found 'state_dict' key")
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("✅ Found 'model_state_dict' key")
    else:
        # Assume the whole dict is state_dict
        state_dict = checkpoint
        print("⚠️ No explicit state_dict key, using entire checkpoint")
    
    # Extract other info
    epoch = checkpoint.get('epoch', 'Unknown')
    best_val_acc = checkpoint.get('best_val_acc', None)
    class_names = checkpoint.get('class_names', CLASS_NAMES)
    model_type = checkpoint.get('model_type', MODEL_TYPE)
    
    print(f"\n📊 Checkpoint Info:")
    print(f"  Epoch: {epoch}")
    print(f"  Best Val Acc: {best_val_acc:.2f}%" if best_val_acc else "  Best Val Acc: Not available")
    print(f"  Classes: {class_names}")
    print(f"  Model Type: {model_type}")
    
else:
    print("⚠️ Checkpoint is a full model, extracting state_dict...")
    state_dict = checkpoint.state_dict()
    class_names = CLASS_NAMES
    model_type = MODEL_TYPE
    best_val_acc = None

# ==================== CHECK STATE DICT ====================
print(f"\n🔍 State Dict Info:")
print(f"  Total parameters: {len(state_dict)}")

# Show first few and last few keys
keys = list(state_dict.keys())
print(f"\n  First 5 keys:")
for key in keys[:5]:
    print(f"    - {key}: {state_dict[key].shape}")

print(f"\n  Last 5 keys:")
for key in keys[-5:]:
    print(f"    - {key}: {state_dict[key].shape}")

# Check if head exists
head_keys = [k for k in keys if 'head' in k]
if head_keys:
    print(f"\n  ✅ Found head layers:")
    for key in head_keys:
        print(f"    - {key}: {state_dict[key].shape}")
else:
    print("\n  ⚠️ No head layers found")

# ==================== CREATE CLEAN VERSION ====================
print(f"\n💾 Creating clean checkpoint...")

clean_checkpoint = {
    'state_dict': state_dict,
    'class_names': class_names,
    'model_type': model_type,
    'num_classes': len(class_names)
}

# Add validation accuracy if available
if best_val_acc is not None:
    clean_checkpoint['val_acc'] = best_val_acc

# ==================== SAVE CLEAN VERSION ====================
print(f"\n💾 Saving clean checkpoint to: {CLEAN_PATH}")
torch.save(clean_checkpoint, CLEAN_PATH)

# ==================== VERIFY SAVED FILE ====================
print(f"\n🔍 Verifying saved file...")
loaded = torch.load(CLEAN_PATH, map_location='cpu')

print(f"✅ Clean checkpoint saved successfully!")
print(f"\n📦 Clean checkpoint contains:")
for key, value in loaded.items():
    if key == 'state_dict':
        print(f"  - {key}: {len(value)} parameters")
    else:
        print(f"  - {key}: {value}")

# ==================== FILE SIZE COMPARISON ====================
original_size = os.path.getsize(CHECKPOINT_PATH) / (1024 * 1024)
clean_size = os.path.getsize(CLEAN_PATH) / (1024 * 1024)

print(f"\n📊 File Size Comparison:")
print(f"  Original: {original_size:.2f} MB")
print(f"  Clean:    {clean_size:.2f} MB")
print(f"  Difference: {original_size - clean_size:.2f} MB")

# ==================== USAGE INSTRUCTIONS ====================
print("\n" + "=" * 70)
print("✨ Conversion Complete!")
print("=" * 70)
print(f"\n📝 Usage Instructions:")
print(f"\n1. Use this file in your Streamlit app:")
print(f"   MODEL_PATH = '{CLEAN_PATH}'")
print(f"\n2. Load the model:")
print(f"""
   checkpoint = torch.load('{CLEAN_PATH}', map_location=device)
   state_dict = checkpoint['state_dict']
   class_names = checkpoint['class_names']
   
   # Create model and load weights
   model = mobilevig_{model_type}(num_classes=1000)
   model.head = nn.Conv2d(512, {len(class_names)}, 1, bias=True)
   
   model_dict = model.state_dict()
   filtered_dict = {{k: v for k, v in state_dict.items() 
                    if k in model_dict and v.shape == model_dict[k].shape}}
   model_dict.update(filtered_dict)
   model.load_state_dict(model_dict)
""")

print("\n✅ Ready for production use!")
print("=" * 70)