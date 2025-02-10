import streamlit as st
import torch
import torch.nn as nn
import timm
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
import time
from io import BytesIO

# --- Global Configurations ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Definitions ---

class SimpleDecoder(nn.Module):
    """
    A simplified UNet-like decoder without attention blocks.
    Uses transposed convolutions and skip connections.
    """
    def __init__(self, encoder_channels, out_channels=1):
        super(SimpleDecoder, self).__init__()
        self.conv_f4 = nn.Conv2d(encoder_channels[3], 512, kernel_size=1)
        self.conv_f3 = nn.Conv2d(encoder_channels[2], 256, kernel_size=1)
        self.conv_f2 = nn.Conv2d(encoder_channels[1], 128, kernel_size=1)
        self.conv_f1 = nn.Conv2d(encoder_channels[0], 64,  kernel_size=1)
        
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.fuse1 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.fuse3 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=1)
    
    def forward(self, features):
        # features: [f1, f2, f3, f4]
        f1, f2, f3, f4 = features
        f4 = self.conv_f4(f4)
        f3 = self.conv_f3(f3)
        f2 = self.conv_f2(f2)
        f1 = self.conv_f1(f1)
        
        x = f4
        x = self.up1(x)
        x = self.fuse1(torch.cat([x, f3], dim=1))
        
        x = self.up2(x)
        x = self.fuse2(torch.cat([x, f2], dim=1))
        
        x = self.up3(x)
        x = self.fuse3(torch.cat([x, f1], dim=1))
        
        x = self.up4(x)
        x = self.up5(x)
        x = self.out_conv(x)
        return x

class SwinTransformerSegModel(nn.Module):
    """
    Full segmentation model with a Swin Transformer encoder and custom decoder.
    """
    def __init__(self, backbone_name="swin_tiny_patch4_window7_224", out_channels=1):
        super(SwinTransformerSegModel, self).__init__()
        self.encoder = timm.create_model(backbone_name, pretrained=True, features_only=True)
        encoder_channels = self.encoder.feature_info.channels()  # e.g., [96, 192, 384, 768]
        self.decoder = SimpleDecoder(encoder_channels, out_channels)
    
    def forward(self, x):
        features = self.encoder(x)
        permuted_features = []
        for f in features:
            if f.dim() == 4 and f.shape[1] < f.shape[-1]:
                f = f.permute(0, 3, 1, 2)
            permuted_features.append(f)
        seg_map = self.decoder(permuted_features)
        return seg_map

# --- Caching and Model Loading ---

@st.cache_resource
def load_model(model_path="swin_segmentation_model.pth"):
    """
    Loads the segmentation model and weights.
    Cached so the model is loaded only once.
    """
    model = SwinTransformerSegModel().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
    return model

# --- Preprocessing and Inference Functions ---

def preprocess_image(image, image_size=(224, 224)):
    """
    Resize and normalize the input image.
    """
    transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return tensor

def overlay_mask_on_image(image, mask, alpha=0.5, color=(0, 255, 0)):
    """
    Overlays a binary mask on an image.
    """
    overlay = image.copy()
    overlay[mask > 0] = color
    blended = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return blended

def predict(model, image, threshold=0.5):
    """
    Runs inference on the input image and returns the predicted binary mask.
    """
    input_tensor = preprocess_image(image)
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output = model(input_tensor)
    prob = torch.sigmoid(output)
    mask = (prob > threshold).float().cpu().numpy().squeeze()  # shape: (H, W)
    return mask

# --- Streamlit App Layout and UI ---

st.title("Chest X-Ray Segmentation Deployment")
st.markdown(
    """
This advanced demo deploys a Swin Transformer-based segmentation model using Streamlit.  
Upload a chest X-Ray image to see the segmentation overlay and mask download options.  
Adjust the segmentation threshold and overlay blending from the sidebar.
"""
)

# Sidebar configuration
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Segmentation Threshold", 0.0, 1.0, 0.5, 0.01)
alpha = st.sidebar.slider("Overlay Alpha", 0.0, 1.0, 0.5, 0.01)
model_path = st.sidebar.text_input("Model Path", "swin_segmentation_model.pth")

# Load the model (cached)
model = load_model(model_path)

# File uploader for chest X-Ray image
uploaded_file = st.file_uploader("Upload a Chest X-Ray image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Error opening image: {e}")
        st.stop()

    # Resize image for display (model expects 224x224)
    display_image = image.resize((224, 224))
    st.image(display_image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Run Segmentation"):
        with st.spinner("Running inference..."):
            start_time = time.time()
            mask = predict(model, image, threshold=threshold)
            inference_time = time.time() - start_time
        
        st.success(f"Inference completed in {inference_time:.2f} seconds")
        
        # Prepare images for display
        display_np = np.array(display_image)
        overlay_img = overlay_mask_on_image(display_np, mask, alpha=alpha, color=(0, 255, 0))
        mask_img = (mask * 255).astype(np.uint8)
        
        st.subheader("Segmentation Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(display_np, caption="Original (224x224)", use_container_width=True)
        with col2:
            st.image(mask_img, caption="Segmentation Mask", use_container_width=True)
        with col3:
            st.image(overlay_img, caption="Overlay (Prediction in Green)", use_container_width=True)
        
        # Download button for the overlay image
        overlay_bgr = cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)
        success_overlay, buffer_overlay = cv2.imencode('.png', overlay_bgr)
        if success_overlay:
            overlay_bytes = buffer_overlay.tobytes()
            st.download_button(
                label="Download Overlay Image",
                data=overlay_bytes,
                file_name="overlay.png",
                mime="image/png"
            )
        
        # Download button for the mask image
        success_mask, buffer_mask = cv2.imencode('.png', mask_img)
        if success_mask:
            mask_bytes = buffer_mask.tobytes()
            st.download_button(
                label="Download Mask Image",
                data=mask_bytes,
                file_name="mask.png",
                mime="image/png"
            )
