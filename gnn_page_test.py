import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import importlib.util

# Set page config
st.set_page_config(
    page_title="MobileViG Image Classification",
    page_icon="🖼️",
    layout="wide"
)

# Configuration
CLASS_NAMES = ["Dark", "Green", "Light", "Medium"]
MODEL_PATH = "mobilevig_finetuned_clean.pth"
MOBILEVIG_PY_PATH = "model/mobilevig.py"

# Title
st.title("🖼️ MobileViG Image Classification")
st.markdown("**Model:** MobileViG-Ti | **Classes:** Dark, Green, Light, Medium")

# Define transforms
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@st.cache_resource
def load_mobilevig_model():
    """Load MobileViG model"""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Import MobileViG module
        if not os.path.exists(MOBILEVIG_PY_PATH):
            st.error(f"❌ MobileViG file not found at: {MOBILEVIG_PY_PATH}")
            return None, None
        
        spec = importlib.util.spec_from_file_location("mobilevig_module", MOBILEVIG_PY_PATH)
        mobilevig = importlib.util.module_from_spec(spec)
        sys.modules['mobilevig_module'] = mobilevig
        spec.loader.exec_module(mobilevig)
        
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ Model weights not found at: {MODEL_PATH}")
            return None, None
        
        # Load checkpoint first
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        
        # Extract state_dict
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict) and not hasattr(checkpoint, 'eval'):
            state_dict = checkpoint
        else:
            # It's a full model
            model = checkpoint
            model.to(device)
            model.eval()
            return model, device
        
        # Create model with 1000 classes first (ImageNet pretrained structure)
        model = mobilevig.mobilevig_s(num_classes=1000)
        
        # Change head to 4 classes
        in_features = 512  # MobileViG-Ti embedding dimension
        model.head = nn.Conv2d(in_features, len(CLASS_NAMES), kernel_size=1, bias=True)
        if hasattr(model, 'dist_head'):
            model.dist_head = nn.Conv2d(in_features, len(CLASS_NAMES), kernel_size=1, bias=True)
        
        # Load weights that match (including fine-tuned head if available)
        model_dict = model.state_dict()
        filtered_dict = {
            k: v for k, v in state_dict.items() 
            if k in model_dict and v.shape == model_dict[k].shape
        }
        
        # Update model dict with filtered weights
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        
        # Log what was loaded
        loaded_keys = set(filtered_dict.keys())
        missing_keys = set(model_dict.keys()) - loaded_keys
        
        if missing_keys:
            st.sidebar.warning(f"⚠️ {len(missing_keys)} layers initialized randomly (including head if not fine-tuned)")
        else:
            st.sidebar.success(f"✅ All weights loaded from checkpoint")
        
        model.to(device)
        model.eval()
        
        return model, device
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        import traceback
        with st.expander("Show Error Details"):
            st.code(traceback.format_exc())
        return None, None

def predict_image(model, image, device):
    """Make prediction on uploaded image"""
    try:
        # Transform image
        img_tensor = test_transforms(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            output = model(img_tensor)
        
        # Calculate probabilities
        probabilities = torch.softmax(output, dim=1)[0] * 100
        
        # Get predicted class
        _, predicted_idx = torch.max(output, 1)
        predicted_class = CLASS_NAMES[predicted_idx.item()]
        
        # Create results dictionary
        results = {
            'predicted_class': predicted_class,
            'predicted_idx': predicted_idx.item(),
            'probabilities': probabilities.cpu().numpy()
        }
        
        return results
    
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        return None

# Sidebar info
with st.sidebar:
    st.header("📊 Model Information")
    st.markdown(f"""
    **Architecture:** MobileViG-Ti  
    **Classes:** {len(CLASS_NAMES)}
    - 🌑 Dark
    - 🟢 Green
    - ☀️ Light
    - 🌤️ Medium
    
    **Model File:** `{MODEL_PATH}`  
    **Architecture File:** `{MOBILEVIG_PY_PATH}`
    """)
    
    st.markdown("---")
    
    # Load model button
    if st.button("🔄 Load Model", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

# Load model
with st.spinner("🔄 Loading MobileViG model..."):
    model, device = load_mobilevig_model()

if model is not None:
    st.sidebar.success("✅ Model loaded successfully!")
    st.sidebar.info(f"🖥️ Device: **{device}**")
else:
    st.sidebar.error("❌ Model not loaded")
    st.error("⚠️ Please check the model and file paths")
    st.stop()

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image to classify into Dark, Green, Light, or Medium"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_container_width=True)

with col2:
    st.subheader("🎯 Prediction Results")
    
    if uploaded_file is not None:
        # Make prediction
        with st.spinner('🔮 Making prediction...'):
            results = predict_image(model, image, device)
        
        if results:
            # Display predicted class with emoji
            class_emojis = {
                "Dark": "🌑",
                "Green": "🟢",
                "Light": "☀️",
                "Medium": "🌤️"
            }
            
            predicted_emoji = class_emojis.get(results['predicted_class'], "🎯")
            
            st.success(f"### {predicted_emoji} **{results['predicted_class']}**")
            st.metric(
                label="Confidence",
                value=f"{results['probabilities'][results['predicted_idx']]:.2f}%"
            )
            
            # Create confidence dataframe
            confidence_data = []
            for i, class_name in enumerate(CLASS_NAMES):
                confidence_data.append({
                    'Class': f"{class_emojis[class_name]} {class_name}",
                    'Confidence (%)': f"{results['probabilities'][i]:.2f}"
                })
            
            confidence_df = pd.DataFrame(confidence_data)
            confidence_df = confidence_df.sort_values(
                'Confidence (%)', 
                ascending=False,
                key=lambda x: x.str.rstrip('%').astype(float)
            )
            
            st.markdown("---")
            st.subheader("📊 All Confidence Scores")
            st.dataframe(
                confidence_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Visualization
            st.markdown("---")
            st.subheader("📈 Confidence Distribution")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Colors for each class
            class_colors = {
                "Dark": "#2c3e50",
                "Green": "#27ae60",
                "Light": "#f39c12",
                "Medium": "#3498db"
            }
            
            colors = [class_colors[class_name] if i != results['predicted_idx'] 
                     else '#e74c3c' for i, class_name in enumerate(CLASS_NAMES)]
            
            bars = ax.barh(CLASS_NAMES, results['probabilities'], color=colors, alpha=0.8)
            
            # Highlight predicted class
            bars[results['predicted_idx']].set_edgecolor('#c0392b')
            bars[results['predicted_idx']].set_linewidth(3)
            
            ax.set_xlabel('Confidence (%)', fontsize=12, fontweight='bold')
            ax.set_title('Prediction Confidence by Class', fontsize=14, fontweight='bold', pad=20)
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            ax.set_xlim(0, 100)
            
            # Add percentage labels
            for i, v in enumerate(results['probabilities']):
                ax.text(v + 2, i, f'{v:.2f}%', va='center', fontsize=11, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            
    else:
        st.info("👆 Upload an image to see predictions")
        
        # Show example
        st.markdown("---")
        st.markdown("### 💡 Example Classes")
        example_cols = st.columns(4)
        
        class_descriptions = {
            "Dark": "🌑 Very dark/dim lighting conditions",
            "Green": "🟢 Green-tinted or natural lighting",
            "Light": "☀️ Bright/well-lit conditions",
            "Medium": "🌤️ Moderate/balanced lighting"
        }
        
        for i, (class_name, desc) in enumerate(class_descriptions.items()):
            with example_cols[i]:
                st.markdown(f"**{class_name}**")
                st.caption(desc)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <p>Built with Streamlit | Powered by MobileViG & PyTorch</p>
    <p style='font-size: 0.8em;'>Vision GNN: An Image is Worth Graph of Nodes (2022)</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Instructions in expander
with st.expander("ℹ️ How to Use This App"):
    st.markdown("""
    ### 📖 Instructions
    
    1. **Check Model Status** - Verify that the model is loaded (green checkmark in sidebar)
    2. **Upload Image** - Click "Browse files" or drag & drop an image
    3. **View Results** - See the predicted class and confidence scores
    
    ### 📁 Required Files
    
    - `m_MobileViG_ti_50_epoch_0.5_d.pth` - Trained model weights
    - `model/mobilevig.py` - MobileViG architecture file
    - `timm` library installed (`pip install timm`)
    
    ### 🎯 Classes
    
    - **Dark** 🌑 - Very dark or dim lighting
    - **Green** 🟢 - Green-tinted lighting
    - **Light** ☀️ - Bright lighting
    - **Medium** 🌤️ - Moderate lighting
    
    ### ⚙️ Model Details
    
    - **Architecture:** MobileViG-Ti (Vision Graph Neural Network)
    - **Input Size:** 224×224 pixels
    - **Training:** 50 epochs with 0.5 dropout
    - **Preprocessing:** ImageNet normalization
    """)

# Troubleshooting section
with st.expander("🔧 Troubleshooting"):
    st.markdown("""
    ### Common Issues
    
    **❌ Model file not found**
    - Check that `m_MobileViG_ti_50_epoch_0.5_d.pth` exists in the current directory
    - Or update `MODEL_PATH` variable at the top of the script
    
    **❌ MobileViG file not found**
    - Check that `mobilevig.py` exists in the `model/` folder
    - Or update `MOBILEVIG_PY_PATH` variable at the top of the script
    
    **❌ Import error**
    - Install required packages:
      ```bash
      pip install streamlit torch torchvision timm pandas matplotlib pillow
      ```
    
    **❌ CUDA/GPU issues**
    - The app will automatically use CPU if CUDA is not available
    - For GPU support, install: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
    """)